import chromadb
from backend.config import DB_PATH
from backend.embeddings import embed_query

# Persistent client pointing to my_chroma_db
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Collections for texts and images
text_collection = chroma_client.get_or_create_collection(name="text_collection")
image_collection = chroma_client.get_or_create_collection(name="image_collection")

def retrieve_text_contents(query: str, top_n: int = 5):
    """Retrieve text documents from text_collection matching the query."""
    query_embedding = embed_query(query)
    return text_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "metadatas"]
    )

def retrieve_image_contents(query: str, top_n: int = 3):
    """Retrieve image metadata matching the query."""
    query_embedding = embed_query(query)
    return image_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["metadatas"]
    )

def clear_collections():
    """Clear all data from both collections."""
    # ChromaDB collections can be cleared by deleting and recreating or dropping all items.
    # To delete all items, we can fetch all ids and delete them.
    text_data = text_collection.get()
    if text_data and text_data["ids"]:
        text_collection.delete(ids=text_data["ids"])
        
    image_data = image_collection.get()
    if image_data and image_data["ids"]:
        image_collection.delete(ids=image_data["ids"])
