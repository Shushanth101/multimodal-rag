import os
import sys
import json

# Add parent directory of backend (workspace root) to Python path
backend_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_parent_dir not in sys.path:
    sys.path.insert(0, backend_parent_dir)

import base64
import shutil
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from backend.config import BASE_DIR, UPLOAD_DIR, IMAGE_DIR
from backend.database import text_collection, image_collection, clear_collections
from backend.pdf_processor import process_pdf
from backend.embeddings import embed_texts, embed_images
from backend.agent import agent
from backend.schemas import QueryRequest, QueryResponse, SourceResponse

app = FastAPI(title="Multimodal RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif"
}

@app.post("/api/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    """Uploads and ingests PDF files into ChromaDB."""
    ingested_stats = []
    
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
            
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            # Process PDF to get text chunks and images
            chunked_docs, images = process_pdf(file_path, extract_images=True)
            
            # 1. Embed and Add Texts
            if chunked_docs:
                texts = [doc.page_content for doc in chunked_docs]
                text_embeddings = embed_texts(texts)
                text_collection.add(
                    ids=[f"{file.filename}_chunk_{i}" for i in range(len(texts))],
                    documents=texts,
                    embeddings=text_embeddings,
                    metadatas=[
                        {
                            "source": file_path,
                            "page": doc.metadata.get("page", -1),
                            "chunk_index": i
                        }
                        for i, doc in enumerate(chunked_docs)
                    ]
                )
                
            # 2. Embed and Add Images
            if images:
                image_embeddings = embed_images(images)
                image_collection.add(
                    ids=[f"{file.filename}_img_{i}" for i in range(len(images))],
                    embeddings=image_embeddings,
                    metadatas=[
                        {
                            "source": file_path,
                            "path": img["path"],
                            "page": img["page"],
                            "extension": img["extension"]
                        }
                        for i, img in enumerate(images)
                    ]
                )
                
            ingested_stats.append({
                "filename": file.filename,
                "chunks": len(chunked_docs),
                "images": len(images)
            })
            
        except Exception as e:
            # Clean up saved file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}: {str(e)}")
            
    return {"status": "success", "ingested": ingested_stats}

@app.post("/api/query")
async def query_rag(request: QueryRequest):
    """Queries the agentic LangGraph workflow, streaming text tokens and sending sources at the end."""
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "model_name": request.model_name
        }
    }
    
    async def event_generator():
        try:
            # Stream AI text tokens chunk-by-chunk using agent.stream
            for chunk, metadata in agent.stream(
                {"messages": [HumanMessage(content=request.question)], "sources": []},
                config=config,
                stream_mode="messages"
            ):
                if hasattr(chunk, "content") and isinstance(chunk.content, str) and chunk.content:
                    if metadata.get("langgraph_node") == "llm_call":
                        yield f"data: {json.dumps({'type': 'token', 'text': chunk.content})}\n\n"
            
            # Fetch final state to retrieve all accumulated sources
            final_state = agent.get_state(config)
            raw_sources = final_state.values.get("sources", [])
            
            seen = set()
            sources_response = []
            
            for s in raw_sources:
                key = (s["source"], s["page"], s["type"])
                if key in seen:
                    continue
                seen.add(key)
                
                entry = {
                    "type": s["type"],
                    "source": os.path.basename(s["source"]),
                    "page": s["page"]
                }
                
                if s["type"] == "image":
                    path = s.get("path")
                    if path and os.path.exists(path):
                        ext = path.rsplit(".", 1)[-1].lower()
                        try:
                            with open(path, "rb") as img_file:
                                entry["b64"] = base64.b64encode(img_file.read()).decode("utf-8")
                            entry["mime"] = MIME_TYPES.get(ext, "image/jpeg")
                            entry["path"] = path
                        except Exception as e:
                            print(f"Could not encode source image: {e}")
                            
                sources_response.append(entry)
                
            # Send final sources payload
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_response})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/files")
async def list_files():
    """Lists files currently in the database with their metrics."""
    # Get all items from text_collection
    text_data = text_collection.get(include=["metadatas"])
    image_data = image_collection.get(include=["metadatas"])
    
    files_info = {}
    
    # Process text metadata
    if text_data and text_data["metadatas"]:
        for meta in text_data["metadatas"]:
            source = meta.get("source", "Unknown")
            fname = os.path.basename(source)
            if fname not in files_info:
                files_info[fname] = {"filename": fname, "chunks": 0, "images": 0}
            files_info[fname]["chunks"] += 1
            
    # Process image metadata
    if image_data and image_data["metadatas"]:
        for meta in image_data["metadatas"]:
            source = meta.get("source", "Unknown")
            fname = os.path.basename(source)
            if fname not in files_info:
                files_info[fname] = {"filename": fname, "chunks": 0, "images": 0}
            files_info[fname]["images"] += 1
            
    return list(files_info.values())

@app.delete("/api/files")
async def delete_all_files():
    """Deletes all vector database entries and physical files."""
    try:
        # Clear vector database collections
        clear_collections()
        
        # Clear physical folders
        for folder in [UPLOAD_DIR, IMAGE_DIR]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
                    
        return {"status": "success", "message": "All vectors and document files successfully deleted."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear workspace: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Mount frontend directory for SPA static assets
frontend_dir = os.path.join(BASE_DIR, "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

@app.get("/")
async def serve_frontend():
    """Serves the main SPA index.html."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend UI file not found. Place index.html inside the frontend directory."}

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Add parent directory of backend (workspace root) to Python path
    backend_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_parent_dir not in sys.path:
        sys.path.insert(0, backend_parent_dir)
        
    # Set PYTHONPATH environment variable so uvicorn reloader subprocesses inherit it
    os.environ["PYTHONPATH"] = backend_parent_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    
    # Run uvicorn dynamically based on directory
    if os.path.basename(os.getcwd()) == "backend":
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    else:
        uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

