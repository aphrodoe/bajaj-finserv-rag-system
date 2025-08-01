import os
import requests
import uuid
import urllib.parse
# **1. Import the new library**
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import time
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys. Set GOOGLE_API_KEY and PINECONE_API_KEY as environment variables.")

# --- Initialize Clients ---
genai.configure(api_key=GOOGLE_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

print("--- Starting PDF Processing ---")
print("Successfully initialized Google AI and Pinecone clients.")

def process_and_store_document(url: str):
    try:
        print(f"Downloading document from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()

        parsed = urllib.parse.urlparse(url)
        file_name = os.path.basename(parsed.path)
        file_name = file_name.replace(" ", "_")
        file_path = os.path.join(".", f"temp_{file_name}")

        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Document saved temporarily to {file_path}")

        # --- Extract text from PDF using PyMuPDF ---
        text = ""
        if file_path.lower().endswith(".pdf"):
            # **2. Use fitz (PyMuPDF) to open and read the PDF**
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        else:
            raise ValueError("Only PDF files are supported in this version.")

        # Optional: Check if the text was extracted correctly
        if not text.strip():
            print("Warning: No text was extracted from the PDF. The document might be an image or have a complex format.")
            os.remove(file_path)
            return

        print(f"Successfully extracted {len(text)} characters of text from the document.")

        # --- Chunk text ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        print(f"Text split into {len(chunks)} chunks.")

        # --- Embed and Store in Batches ---
        index_name = "hackrx-gemini-index"
        dimension = 768
        
        if index_name not in pinecone.list_indexes().names():
            print(f"Creating new Pinecone index: {index_name} with dimension {dimension}")
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)

        index = pinecone.Index(index_name)

        batch_size = 50
        print(f"Embedding and storing chunks in batches of {batch_size}...")

        for i in range(0, len(chunks), batch_size):
            chunk_batch = chunks[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}...")

            result = genai.embed_content(
                model="models/embedding-001",
                content=chunk_batch,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            embeddings = result['embedding']
            
            vectors_to_upsert = []
            for j, chunk in enumerate(chunk_batch):
                vectors_to_upsert.append({
                    "id": str(uuid.uuid4()),
                    "values": embeddings[j],
                    "metadata": {"text": chunk}
                })
            
            index.upsert(vectors=vectors_to_upsert)
            time.sleep(1)

        print("All document chunks embedded and stored in Pinecone using Gemini.")
        os.remove(file_path)
        print("Temporary file deleted.")

    except Exception as e:
        import traceback
        from dotenv import load_dotenv
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    process_and_store_document(document_url)