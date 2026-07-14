# === Cell 0 ===
# Install dependencies
!pip install -qU langchain-pymupdf4llm pymupdf langchain-text-splitters langchain-groq langgraph google-genai chromadb pydantic


# === Cell 1 ===
# Standard Library
import os
import io
import time
import operator
from typing import Union, Literal
from typing_extensions import TypedDict, Annotated

# PDF/Image Processing
import fitz
from PIL import Image

# Google GenAI
from google import genai
from google.genai import types

# LangChain / LangGraph
from langchain.tools import tool
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import (
    ToolMessage,
    SystemMessage,
    HumanMessage,
    AnyMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Vector DB
import chromadb


# === Cell 2 ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)




def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):

    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]

            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            filename = f"{pdf_name}_page{page_num+1}_img{img_index+1}.{image_ext}"
            filepath = os.path.join(output_folder, filename)



            with open(filepath, "wb") as f:
                f.write(image_bytes)


            pil_image = Image.open(io.BytesIO(image_bytes))


            images.append({
                "path": filepath,
                "image_bytes":image_bytes,
                "page": page_num,
                "extension": image_ext,
                "index": img_index
            })


            print(f"Saved: {filepath}")


    doc.close()

    print(f"Extraction complete. Total images saved: {len(images)}")

    return images



def process_pdf(path:str,
                extract_images=True,
                *,
                images_output_folder="/content/images"):

    loader = PyMuPDF4LLMLoader(
        path,
        mode="page"
    )

    docs = loader.load()

    chunked_documents = text_splitter.split_documents(docs)

    images = []

    if extract_images:
        images = extract_images_from_pdf(
            path,
            images_output_folder
        )
        return chunked_documents, images


    return chunked_documents

# === Cell 3 ===
from google import genai
from google.colab import userdata
from google.genai import types

mime_type = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif"
}


os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')
client = genai.Client()

# === Cell 4 ===
import time

def embed_texts(texts: list[str]):
    embeddings = []

    for i in range(0, len(texts), 8):
        chunk = texts[i:i + 8]

        contents = [
            types.Content(parts=[types.Part(text=t)])
            for t in chunk
        ]

        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=contents,
            config={
                "output_dimensionality": 768
            }
        )

        embeddings.extend([e.values for e in result.embeddings])

        time.sleep(1)

    return embeddings

def embed_images(images: list[dict]):
  embeddings=[]
  for i in range(0,len(images),6):
      image_chunks = images[i:i+6]
      contents = [
      types.Content(
          parts=[
              types.Part.from_bytes(
                  data=chunk["image_bytes"],
                  mime_type=mime_type[chunk["extension"]]
              )
          ]
      )
      for chunk in image_chunks
      ]
      result = client.models.embed_content(
          model="gemini-embedding-2",
          contents=contents,
          config={
              'output_dimensionality': 768
          }
      )
      embeddings.extend([e.values for e in result.embeddings])
      time.sleep(1)
  return embeddings




def embed_query(query:str):
  result = client.models.embed_content(
      model="gemini-embedding-2",
      contents=f"task: question answering | query: {query}",
      config={
          'output_dimensionality': 768
      }
  )
  return result.embeddings[0].values



# === Cell 5 ===
import chromadb

chroma_client = chromadb.PersistentClient(path="./my_chroma_db")

# === Cell 6 ===
text_collection = chroma_client.get_or_create_collection(name="text_collection")
image_collection = chroma_client.get_or_create_collection(name="image_collection")

# === Cell 7 ===
def ingest(file_paths: list[str]):
    for path in file_paths:
        chunked_documents, images = process_pdf(path)

        texts = [doc.page_content for doc in chunked_documents]
        text_embeddings = embed_texts(texts)

        text_collection.add(
            ids=[f"{path}_chunk_{i}" for i in range(len(texts))],
            documents=texts,
            embeddings=text_embeddings,
            metadatas=[
                {
                    "source": path,
                    "page": doc.metadata.get("page", -1),
                    "chunk_index": i
                }
                for i, doc in enumerate(chunked_documents)
            ]
        )

        image_embeddings = embed_images(images)

        image_collection.add(
            ids=[f"{path}_img_{i}" for i in range(len(images))],
            embeddings=image_embeddings,
            metadatas=[
                {
                    "source": path,
                    "path": img["path"],
                    "page": img["page"],
                    "extension": img["extension"],
                }
                for i, img in enumerate(images)
            ]
        )

        print(f"Ingested: {len(texts)} text chunks, {len(images)} images from {path}")

# === Cell 8 ===
def retrieve_text_contents(query: str, top_n: int = 5):
    query_embedding = embed_query(query)
    return text_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "metadatas"]
    )


def retrieve_image_contents(query: str, top_n: int = 3):
    query_embedding = embed_query(query)
    return image_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["metadatas"]
    )

# === Cell 9 ===
from typing import Union, Literal
from typing_extensions import TypedDict, Annotated
import base64


@tool("retrieve_text_content", description="Retrieves relevant text content from the PDF")
def retrieve_text_content(query: str, config: RunnableConfig, top_n: Union[int, str] = 5):
    top_n = int(top_n)
    text_results = retrieve_text_contents(query, top_n)
    text_docs = text_results["documents"]
    text_metas = text_results["metadatas"]

    full_text_result = ""
    for i in range(len(text_docs[0])):
        doc = text_docs[0][i]
        meta = text_metas[0][i]
        full_text_result += f"\n[Text {i+1}] Page {meta['page']}:\n{doc}\n"

    return full_text_result


@tool("retrieve_image_content", description="Retrieves relevant images from the PDF")
def retrieve_image_content(query: str, config: RunnableConfig, top_n: Union[int, str] = 3):
    top_n = int(top_n)
    image_results = retrieve_image_contents(query, top_n)
    img_metas = image_results["metadatas"]

    b64_images = []
    for meta in img_metas[0]:
        path = meta["path"]
        ext = meta.get("extension", "jpeg")
        try:
            with open(path, "rb") as f:
                b64_images.append({
                    "data": base64.b64encode(f.read()).decode("utf-8"),
                    "mime_type": mime_type.get(ext, "image/jpeg")
                })
        except Exception as e:
            print(f"Could not load image at {path}: {e}")

    return b64_images


# === Cell 10 ===
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

# === Cell 11 ===
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
tools = [retrieve_text_content, retrieve_image_content]
tools_by_name = {t.name: t for t in tools}
model_with_tools = llm.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    sources: Annotated[list[dict], operator.add]



def llm_call(state: dict):
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# def tool_node(state: dict):
#     result = []
#     for tool_call in state["messages"][-1].tool_calls:
#         tool = tools_by_name[tool_call["name"]]

#         if tool_call["name"] == "retrieve_image_content":
#             b64_images = tool.invoke(tool_call["args"])
#             result.append(ToolMessage("Here are the retrieved images", tool_call_id=tool_call["id"]))
#             for img in b64_images:
#                 result.append(HumanMessage(content=[
#                     {"type": "text", "text": "Retrieved image"},
#                     {"type": "image_url", "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"}}
#                 ]))
#         else:
#             observation = tool.invoke(tool_call["args"])
#             result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

#     return {"messages": result}

def tool_node(state: dict):
    result = []
    sources = []

    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]

        if tool_call["name"] == "retrieve_image_content":
            image_results = retrieve_image_contents(tool_call["args"]["query"], int(tool_call["args"].get("top_n", 3)))
            img_metas = image_results["metadatas"][0]

            # collect image sources
            for meta in img_metas:
                sources.append({
                    "type": "image",
                    "source": meta["source"],
                    "page": meta["page"] + 1,
                    "path": meta["path"]
                })

            b64_images = retrieve_image_content.invoke(tool_call["args"])
            result.append(ToolMessage("Here are the retrieved images", tool_call_id=tool_call["id"]))
            for img in b64_images:
                result.append(HumanMessage(content=[
                    {"type": "text", "text": "Retrieved image"},
                    {"type": "image_url", "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"}}
                ]))
        else:
            text_results = retrieve_text_contents(tool_call["args"]["query"], int(tool_call["args"].get("top_n", 5)))
            text_metas = text_results["metadatas"][0]

            # collect text sources
            for meta in text_metas:
                sources.append({
                    "type": "text",
                    "source": meta["source"],
                    "page": meta["page"] + 1,
                    "chunk_index": meta["chunk_index"]
                })

            observation = retrieve_text_content.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"messages": result, "sources": sources}

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return "__end__"


checkpointer = MemorySaver()

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", "__end__"])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile(checkpointer=checkpointer)

# === Cell 12 ===


# === Cell 13 ===
ingest(["/content/1706.03762v7.pdf"])

# === Cell 14 ===
messages = agent.invoke(
    {"messages": [HumanMessage(content="Explain the scaled dot product attention mechanism")]},
    config={"configurable": {"thread_id": "session_001"}}
)

for m in messages["messages"]:
    m.pretty_print()


# === Cell 15 ===
messages = agent.invoke(
    {"messages": [HumanMessage(content="query you retrieve_image_content tool with this query=(add&norm softmax linear feedforward multihead attention) and tell me what diagrams do you see")]},
    config={"configurable": {"thread_id": "session_0002"}}
)

for m in messages["messages"]:
    m.pretty_print()

# === Cell 16 ===
import sys
from IPython.display import display, Image as IPImage, Markdown
import base64

config = {"configurable": {"thread_id": "session_stream_001"}}

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("exit", "quit"):
        print("Exiting...")
        break

    sources = []

    # ── stream the agent ──────────────────────────────────────────────
    print("\nAssistant: ", end="", flush=True)

    for chunk, metadata in agent.stream(
        {"messages": [HumanMessage(content=user_input)], "sources": []},
        config=config,
        stream_mode="messages"
    ):
        # stream AI text tokens
        if hasattr(chunk, "content") and isinstance(chunk.content, str) and chunk.content:
            node = metadata.get("langgraph_node")
            if node == "llm_call":
                print(chunk.content, end="", flush=True)

    print()  # newline after response

    # ── get full state for sources ─────────────────────────────────────
    final_state = agent.get_state(config)
    sources = final_state.values.get("sources", [])

    # ── display sources ────────────────────────────────────────────────
    if sources:
        print("\n")
        display(Markdown("### 📚 Sources Used"))
        seen = set()
        for s in sources:
            key = (s["source"], s["page"], s["type"])
            if key in seen:
                continue
            seen.add(key)

            if s["type"] == "image":
                display(Markdown(f"**🖼️ Image** | `{s['source']}` | Page {s['page']}"))
                try:
                    display(IPImage(filename=s["path"], width=500))
                except Exception as e:
                    print(f"  Could not display image: {e}")
            else:
                display(Markdown(f"**📄 Text** | `{s['source']}` | Page {s['page']}"))

# === Cell 17 ===
from IPython.display import display, HTML
import json, base64, os

def build_chat_gui():
    html = """
<div id="rag-chat">
  <style>
    #rag-chat {
      font-family: 'Segoe UI', sans-serif;
      max-width: 860px;
      margin: 0 auto;
      background: #0f0f0f;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid #2a2a2a;
    }
    #chat-header {
      background: #161616;
      padding: 16px 24px;
      border-bottom: 1px solid #2a2a2a;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    #chat-header span.dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      background: #7c3aed;
      display: inline-block;
      box-shadow: 0 0 8px #7c3aed;
    }
    #chat-header h3 {
      margin: 0;
      color: #e5e5e5;
      font-size: 15px;
      font-weight: 600;
    }
    #chat-header small {
      color: #555;
      font-size: 12px;
      margin-left: auto;
    }
    #chat-window {
      height: 480px;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      background: #0f0f0f;
    }
    #chat-window::-webkit-scrollbar { width: 4px; }
    #chat-window::-webkit-scrollbar-track { background: transparent; }
    #chat-window::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 4px; }
    .msg-row { display: flex; gap: 10px; align-items: flex-start; }
    .msg-row.user { flex-direction: row-reverse; }
    .avatar {
      width: 32px; height: 32px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 14px;
      flex-shrink: 0;
    }
    .avatar.ai { background: #1e1030; color: #a78bfa; border: 1px solid #3b1f6e; }
    .avatar.user { background: #0f2318; color: #4ade80; border: 1px solid #14532d; }
    .bubble {
      max-width: 75%;
      padding: 12px 16px;
      border-radius: 14px;
      font-size: 14px;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .bubble.ai {
      background: #1a1a1a;
      color: #d4d4d4;
      border: 1px solid #2a2a2a;
      border-top-left-radius: 4px;
    }
    .bubble.user {
      background: #1a2e1a;
      color: #d4d4d4;
      border: 1px solid #1f4d1f;
      border-top-right-radius: 4px;
    }
    .sources-block {
      margin-top: 12px;
      border-top: 1px solid #2a2a2a;
      padding-top: 10px;
    }
    .sources-title {
      font-size: 11px;
      color: #666;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }
    .source-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #111;
      border: 1px solid #2a2a2a;
      border-radius: 20px;
      padding: 4px 10px;
      font-size: 12px;
      color: #888;
      margin: 3px 3px 3px 0;
      cursor: pointer;
      transition: border-color 0.2s;
    }
    .source-chip:hover { border-color: #7c3aed; color: #a78bfa; }
    .source-chip .icon { font-size: 13px; }
    .source-img {
      margin-top: 8px;
      border-radius: 8px;
      border: 1px solid #2a2a2a;
      max-width: 100%;
    }
    .typing-dots {
      display: flex; gap: 4px; align-items: center; padding: 4px 0;
    }
    .typing-dots span {
      width: 7px; height: 7px; background: #555;
      border-radius: 50%;
      animation: bounce 1.2s infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.5; }
      40% { transform: translateY(-5px); opacity: 1; }
    }
    #chat-input-area {
      display: flex;
      gap: 10px;
      padding: 14px 20px;
      background: #161616;
      border-top: 1px solid #2a2a2a;
    }
    #chat-input {
      flex: 1;
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 10px;
      padding: 10px 14px;
      color: #e5e5e5;
      font-size: 14px;
      outline: none;
      resize: none;
      font-family: inherit;
      transition: border-color 0.2s;
      height: 42px;
      line-height: 1.4;
    }
    #chat-input:focus { border-color: #7c3aed; }
    #send-btn {
      background: #7c3aed;
      border: none;
      border-radius: 10px;
      width: 42px; height: 42px;
      cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      transition: background 0.2s;
      flex-shrink: 0;
    }
    #send-btn:hover { background: #6d28d9; }
    #send-btn:disabled { background: #2a2a2a; cursor: not-allowed; }
    #send-btn svg { width: 18px; height: 18px; fill: white; }
  </style>

  <div id="chat-header">
    <span class="dot"></span>
    <h3>Multimodal RAG</h3>
    <small id="status-text">Ready</small>
  </div>

  <div id="chat-window">
    <div class="msg-row">
      <div class="avatar ai">✦</div>
      <div class="bubble ai">Hey! Ask me anything about the ingested PDFs — I can retrieve text and diagrams.</div>
    </div>
  </div>

  <div id="chat-input-area">
    <textarea id="chat-input" placeholder="Ask something..." rows="1"></textarea>
    <button id="send-btn">
      <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>
    </button>
  </div>
</div>

<script>
(function() {
  const chatWindow = document.getElementById('chat-window');
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const statusText = document.getElementById('status-text');

  function scrollToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'msg-row user';
    row.innerHTML = `
      <div class="avatar user">U</div>
      <div class="bubble user">${text}</div>
    `;
    chatWindow.appendChild(row);
    scrollToBottom();
  }

  function appendTyping() {
    const row = document.createElement('div');
    row.className = 'msg-row';
    row.id = 'typing-row';
    row.innerHTML = `
      <div class="avatar ai">✦</div>
      <div class="bubble ai"><div class="typing-dots"><span></span><span></span><span></span></div></div>
    `;
    chatWindow.appendChild(row);
    scrollToBottom();
    return row;
  }

  function removeTyping() {
    const el = document.getElementById('typing-row');
    if (el) el.remove();
  }

  function appendAIMessage(text, sources) {
    const row = document.createElement('div');
    row.className = 'msg-row';

    let sourcesHTML = '';
    if (sources && sources.length > 0) {
      const seen = new Set();
      let chips = '';
      let images = '';
      sources.forEach(s => {
        const key = s.source + '|' + s.page + '|' + s.type;
        if (seen.has(key)) return;
        seen.add(key);
        const icon = s.type === 'image' ? '🖼️' : '📄';
        const fname = s.source.split('/').pop();
        chips += `<span class="source-chip"><span class="icon">${icon}</span>${fname} · p${s.page}</span>`;
        if (s.type === 'image' && s.b64) {
          images += `<img class="source-img" src="data:${s.mime};base64,${s.b64}" title="${fname} page ${s.page}" />`;
        }
      });
      sourcesHTML = `
        <div class="sources-block">
          <div class="sources-title">Sources</div>
          ${chips}
          ${images}
        </div>`;
    }

    row.innerHTML = `
      <div class="avatar ai">✦</div>
      <div class="bubble ai">${text}${sourcesHTML}</div>
    `;
    chatWindow.appendChild(row);
    scrollToBottom();
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    sendBtn.disabled = true;
    statusText.textContent = 'Thinking...';
    appendUserMessage(text);
    const typingRow = appendTyping();

    try {
const result = await google.colab.kernel.invokeFunction(
  'run_rag_query', [text], {}
);
const data = JSON.parse(result.data['application/json']);
      removeTyping();
      appendAIMessage(data.answer, data.sources);
      statusText.textContent = 'Ready';
    } catch(e) {
      removeTyping();
      appendAIMessage('Error: ' + e.message, []);
      statusText.textContent = 'Error';
    }

    sendBtn.disabled = false;
    scrollToBottom();
  }

  sendBtn.addEventListener('click', sendMessage);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
})();
</script>
"""
    display(HTML(html))

# ── Python backend called by the GUI ──────────────────────────────────────────
import google.colab.output

config = {"configurable": {"thread_id": "gui_session_001"}}

def run_rag_query(user_input):
    answer_parts = []

    for chunk, metadata in agent.stream(
        {"messages": [HumanMessage(content=user_input)], "sources": []},
        config=config,
        stream_mode="messages"
    ):
        if hasattr(chunk, "content") and isinstance(chunk.content, str) and chunk.content:
            if metadata.get("langgraph_node") == "llm_call":
                answer_parts.append(chunk.content)

    answer = "".join(answer_parts)

    # get sources from final state
    final_state = agent.get_state(config)
    raw_sources = final_state.values.get("sources", [])

    # dedupe + attach b64 for images
    seen = set()
    sources = []
    for s in raw_sources:
        key = (s["source"], s["page"], s["type"])
        if key in seen:
            continue
        seen.add(key)
        entry = {
            "type": s["type"],
            "source": s["source"],
            "page": s["page"],
        }
        if s["type"] == "image":
            try:
                ext = s.get("path", "").rsplit(".", 1)[-1]
                with open(s["path"], "rb") as f:
                    entry["b64"] = base64.b64encode(f.read()).decode()
                entry["mime"] = mime_type.get(ext, "image/jpeg")
                entry["path"] = s["path"]
            except Exception:
                pass
        sources.append(entry)

    return {"application/json": json.dumps({"answer": answer, "sources": sources})}

google.colab.output.register_callback('run_rag_query', run_rag_query)

build_chat_gui()

