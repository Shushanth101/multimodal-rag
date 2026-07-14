from typing import Optional, List
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., description="The query to ask the model about the ingested PDFs.")
    thread_id: str = Field(..., description="Unique session thread ID for LangGraph memory checkpointer.")
    model_name: Optional[str] = Field("gemini-2.5-flash", description="The dynamic model to use for generation (e.g. gemini-2.5-flash, qwen-2.5-32b).")

class SourceResponse(BaseModel):
    type: str = Field(..., description="'text' or 'image'")
    source: str = Field(..., description="Path or filename of the source document.")
    page: int = Field(..., description="Page number of the source document (1-indexed).")
    path: Optional[str] = Field(None, description="Local path of the extracted image if type is image.")
    b64: Optional[str] = Field(None, description="Base64 encoded image string if type is image.")
    mime: Optional[str] = Field(None, description="Image MIME type (e.g. image/png) if type is image.")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The synthesized text answer from the agent.")
    sources: List[SourceResponse] = Field(default=[], description="The list of distinct retrieved sources supporting the answer.")
