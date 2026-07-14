import os
import time
from google import genai
from google.genai import types
from backend.config import GOOGLE_API_KEY, DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSIONALITY

# Initialize Gemini Client lazily to prevent ValueError on import if key is not yet set
client = None

def get_client():
    global client
    if client is None:
        key = GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY is not configured in the environment or .env file.")
        client = genai.Client(api_key=key)
    return client

MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif"
}

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text chunks in batches of 8 using gemini-embedding-2."""
    embeddings = []
    
    for i in range(0, len(texts), 8):
        chunk = texts[i:i + 8]
        contents = [
            types.Content(parts=[types.Part(text=t)])
            for t in chunk
        ]
        
        result = get_client().models.embed_content(
            model=DEFAULT_EMBEDDING_MODEL,
            contents=contents,
            config={
                "output_dimensionality": EMBEDDING_DIMENSIONALITY
            }
        )
        
        embeddings.extend([e.values for e in result.embeddings])
        time.sleep(0.5)  # Slight throttle to avoid rate limits
        
    return embeddings

def embed_images(images: list[dict]) -> list[list[float]]:
    """Embed extracted images in batches of 6 using gemini-embedding-2."""
    embeddings = []
    
    for i in range(0, len(images), 6):
        image_chunks = images[i:i + 6]
        contents = [
            types.Content(
                parts=[
                    types.Part.from_bytes(
                        data=chunk["image_bytes"],
                        mime_type=MIME_TYPES.get(chunk["extension"].lower(), "image/jpeg")
                    )
                ]
            )
            for chunk in image_chunks
        ]
        
        result = get_client().models.embed_content(
            model=DEFAULT_EMBEDDING_MODEL,
            contents=contents,
            config={
                "output_dimensionality": EMBEDDING_DIMENSIONALITY
            }
        )
        
        embeddings.extend([e.values for e in result.embeddings])
        time.sleep(0.5)  # Slight throttle to avoid rate limits
        
    return embeddings

def embed_query(query: str) -> list[float]:
    """Embed the search query with question answering instruction."""
    result = get_client().models.embed_content(
        model=DEFAULT_EMBEDDING_MODEL,
        contents=f"task: question answering | query: {query}",
        config={
            "output_dimensionality": EMBEDDING_DIMENSIONALITY
        }
    )
    return result.embeddings[0].values
