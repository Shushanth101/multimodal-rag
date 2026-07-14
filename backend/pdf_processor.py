import os
import io
import fitz
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from backend.config import IMAGE_DIR

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def extract_images_from_pdf(pdf_path: str, output_folder: str = IMAGE_DIR) -> list[dict]:
    """Extract all images from a PDF file using fitz and save them to the output folder."""
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
            
            # Form clean filename using original PDF name
            filename = f"{pdf_name}_page{page_num+1}_img{img_index+1}.{image_ext}"
            filepath = os.path.join(output_folder, filename)
            
            with open(filepath, "wb") as f:
                f.write(image_bytes)
                
            images.append({
                "path": filepath,
                "image_bytes": image_bytes,
                "page": page_num,
                "extension": image_ext,
                "index": img_index
            })
            print(f"Saved extracted image: {filepath}")
            
    doc.close()
    print(f"Extraction complete. Total images saved: {len(images)}")
    return images

def process_pdf(path: str, extract_images: bool = True) -> tuple[list, list[dict]]:
    """Process PDF to get text chunks and extract images."""
    loader = PyMuPDF4LLMLoader(
        path,
        mode="page"
    )
    docs = loader.load()
    chunked_documents = text_splitter.split_documents(docs)
    
    images = []
    if extract_images:
        images = extract_images_from_pdf(path, IMAGE_DIR)
        
    return chunked_documents, images
