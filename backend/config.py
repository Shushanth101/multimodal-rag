import os
from dotenv import load_dotenv

# Load .env file from the project root if it exists
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Root path of the project (workspace)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_pdfs")
IMAGE_DIR = os.path.join(BASE_DIR, "extracted_images")
DB_PATH = os.path.join(BASE_DIR, "my_chroma_db")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# Default models
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONALITY = 768
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "qwen/qwen3.6-27b"
