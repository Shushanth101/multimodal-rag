import base64
import operator
from typing import Union, Literal
from typing_extensions import TypedDict, Annotated

from langchain.tools import tool
from langchain_core.messages import (
    ToolMessage,
    SystemMessage,
    HumanMessage,
    AnyMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.config import GOOGLE_API_KEY, GROQ_API_KEY
from backend.database import (
    retrieve_text_contents,
    retrieve_image_contents,
)

MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif"
}

@tool("retrieve_text_content", description="Retrieves relevant text content from the PDF")
def retrieve_text_content(query: str, top_n: Union[int, str] = 5) -> str:
    """Retrieves relevant text content from the database."""
    top_n = int(top_n)
    text_results = retrieve_text_contents(query, top_n)
    text_docs = text_results.get("documents", [[]])
    text_metas = text_results.get("metadatas", [[]])
    
    if not text_docs or not text_docs[0]:
        return "No relevant text found."
        
    full_text_result = ""
    for i in range(len(text_docs[0])):
        doc = text_docs[0][i]
        meta = text_metas[0][i]
        full_text_result += f"\n[Text {i+1}] Page {meta.get('page', -1) + 1}:\n{doc}\n"
        
    return full_text_result

@tool("retrieve_image_content", description="Retrieves relevant images from the PDF")
def retrieve_image_content(query: str, top_n: Union[int, str] = 3) -> list[dict]:
    """Retrieves relevant images as base64 strings from the database."""
    top_n = int(top_n)
    image_results = retrieve_image_contents(query, top_n)
    img_metas = image_results.get("metadatas", [[]])
    
    b64_images = []
    if not img_metas or not img_metas[0]:
        return b64_images
        
    for meta in img_metas[0]:
        path = meta.get("path")
        ext = meta.get("extension", "jpeg").lower()
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                b64_images.append({
                    "data": base64.b64encode(f.read()).decode("utf-8"),
                    "mime_type": MIME_TYPES.get(ext, "image/jpeg")
                })
        except Exception as e:
            print(f"Could not load image at {path}: {e}")
            
    return b64_images

SYSTEM_PROMPT = """You are an intelligent research assistant specialized in analyzing and answering questions about PDF documents.

You have access to two retrieval tools:

1. retrieve_text_content
   - Use this for any question that involves textual information: explanations, definitions, methodology, results, conclusions, comparisons, or any factual question about the document.
   - Always use this tool first before retrieve_image_content unless the user is explicitly asking about a figure or diagram.

2. retrieve_image_content
   - Use this when the user asks about figures, diagrams, charts, tables, architectures, or any visual content in the document.
   - You can also use this alongside retrieve_text_content when a visual would help support your answer.

Guidelines:
- Always retrieve relevant content before answering. Never answer from memory alone.
- If the first retrieval does not give enough context, call the tool again with a more specific or rephrased query.
- When answering, clearly reference the page numbers from the retrieved content (e.g., "According to page 3...").
- If text and image results are both retrieved, synthesize them into a single coherent answer.
- If the document does not contain relevant information for the question, clearly say so instead of guessing.
- Keep answers concise but complete. Avoid unnecessary repetition of retrieved text.
- If the user asks a follow-up question, use the conversation history to refine your retrieval query.
"""

# Register tools
tools = [retrieve_text_content, retrieve_image_content]
tools_by_name = {t.name: t for t in tools}

import os

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    sources: Annotated[list[dict], operator.add]

def llm_call(state: dict, config: RunnableConfig):
    # Retrieve configuration for model
    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name", "gemini-2.5-flash")
    
    # Dynamically select generation model
    if model_name.startswith("gemini"):
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY)
    else:
        llm = ChatGroq(model=model_name, groq_api_key=GROQ_API_KEY)
        
    model_with_tools = llm.bind_tools(tools)
    
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def tool_node(state: dict):
    result = []
    sources = []
    
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"messages": result, "sources": sources}
        
    for tool_call in last_msg.tool_calls:
        tool_obj = tools_by_name[tool_call["name"]]
        args = tool_call["args"]
        
        if tool_call["name"] == "retrieve_image_content":
            # Extract meta to track as sources
            top_n = int(args.get("top_n", 3))
            image_results = retrieve_image_contents(args["query"], top_n)
            img_metas = image_results.get("metadatas", [[]])[0]
            
            for meta in img_metas:
                sources.append({
                    "type": "image",
                    "source": meta["source"],
                    "page": meta["page"] + 1,
                    "path": meta["path"]
                })
                
            b64_images = retrieve_image_content.invoke(args)
            result.append(ToolMessage("Here are the retrieved images", tool_call_id=tool_call["id"]))
            
            for img in b64_images:
                result.append(HumanMessage(content=[
                    {"type": "text", "text": "Retrieved image"},
                    {"type": "image_url", "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"}}
                ]))
        else:
            top_n = int(args.get("top_n", 5))
            text_results = retrieve_text_contents(args["query"], top_n)
            text_metas = text_results.get("metadatas", [[]])[0]
            
            for meta in text_metas:
                sources.append({
                    "type": "text",
                    "source": meta["source"],
                    "page": meta["page"] + 1,
                    "chunk_index": meta["chunk_index"]
                })
                
            observation = retrieve_text_content.invoke(args)
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
            
    return {"messages": result, "sources": sources}

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return "__end__"

# Build Graph
checkpointer = MemorySaver()
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", "__end__"])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile(checkpointer=checkpointer)
