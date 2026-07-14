# Multimodal RAG Research Partner

An agentic, multi-modal Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **ChromaDB**, and **LangGraph**. This application processes uploaded PDF documents, extracts text and figures, indexes them in a persistent vector database, and lets you chat with an AI assistant that streams answers and displays retrieved sources and inline diagrams.

---

## Key Features

- **Multi-Modal Retrieval**: Indexes text chunks and extracts/indexes figures from PDF pages using layout-aware PyMuPDF loader.
- **Agentic Workflow**: Managed by a LangGraph StateGraph agent that uses tools to fetch text and image references and reasons over them.
- **Dynamic Model Selection**: Toggle dynamically between `gemini-2.5-flash` (via Google GenAI) and `qwen/qwen3.6-27b` (via Groq) directly in the UI.
- **Real-Time Streaming**: Stream tokens chunk-by-chunk using Server-Sent Events (SSE) for instant conversational responses.
- **Sources & Diagrams inline**: Highlights citations below the AI response showing page numbers, and renders retrieved figures inline with preview modals.
- **Simple Directory Structure**: Segregated clean layout with `backend/` and `frontend/` directories.

---

## Directory Layout

```
├── backend/                # FastAPI application
│   ├── config.py           # Configuration variables and paths
│   ├── database.py         # Persistent ChromaDB client & collections
│   ├── embeddings.py       # Google GenAI embedding wrappers
│   ├── pdf_processor.py    # PyMuPDF parser and image extractor
│   ├── agent.py            # LangGraph agent definitions & tools
│   ├── schemas.py          # Request and response validation schemas
│   └── main.py             # Endpoint routing & static file server
├── frontend/               # Single Page Application UI
│   ├── index.html          # Chat interface structure
│   ├── index.css           # Premium dark UI theme
│   └── index.js            # Ingestion, streaming parser & modal logic
├── requirements.txt        # Virtual environment dependencies
├── .gitignore              # Standard git exclusion configurations
├── uploaded_pdfs/          # Local storage for uploaded PDF files
├── extracted_images/       # Local storage for extracted PDF images
└── my_chroma_db/           # Local folder for ChromaDB database files
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd multimodal-rag
```

### 2. Set up the virtual environment
Create a virtual environment and install the required dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API keys
Create a `.env` file in the root directory and add your keys:
```env
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## Running the Server

Start the FastAPI application directly using `python` inside your virtual environment.

From the **workspace root**:
```powershell
.venv\Scripts\python backend/main.py
```

Or from inside the **backend directory**:
```powershell
cd backend
..\.venv\Scripts\python main.py
```

Once running, open your web browser and navigate to:
```
http://127.0.0.1:8000
```

---

## Usage Guide

1. **Ingest Documents**: In the left sidebar, click the upload box (or drag & drop a PDF) to upload papers. The system will slice the text and extract all images.
2. **Track Ingestion**: The sidebar list will update to show the file name, extracted text chunks, and image counts.
3. **Select a Model**: Choose either `Gemini 2.5 Flash` or `Qwen 3.6 27B (Groq)` in the Model Selection dropdown.
4. **Chat**: Enter your prompt in the chat input (e.g. *"Show me the diagram representing the architecture"* or *"Explain scaled dot product attention"*). Watch the token generation stream live, and click inline images to zoom.
5. **Reset database**: Click **Clear All** next to "Ingested Files" to delete all vector embeddings and uploaded files.
