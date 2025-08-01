import os
import requests
import uuid
import urllib.parse
import fitz 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import time
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    
    def __init__(self, google_api_key: Optional[str] = None, pinecone_api_key: Optional[str] = None):

        load_dotenv()
        
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        
        if not self.google_api_key or not self.pinecone_api_key:
            raise ValueError("Missing API keys. Set GOOGLE_API_KEY and PINECONE_API_KEY as environment variables.")
        
        genai.configure(api_key=self.google_api_key)
        self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        
        self.current_index_name = None 
        self.dimension = 768
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.batch_size = 50
        
        logger.info("Successfully initialized Google AI and Pinecone clients.")
    
    def download_document(self, url: str) -> str:

        logger.info(f"Downloading document from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()

        parsed = urllib.parse.urlparse(url)
        file_name = os.path.basename(parsed.path)
        file_name = file_name.replace(" ", "_")
        file_path = os.path.join(".", f"temp_{file_name}")

        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Document saved temporarily to {file_path}")
        return file_path
    
    def extract_text_from_pdf(self, file_path: str) -> str:

        text = ""
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        else:
            raise ValueError("Only PDF files are supported in this version.")
        
        if not text.strip():
            logger.warning("No text was extracted from the PDF. The document might be an image or have a complex format.")
            return ""
        
        logger.info(f"Successfully extracted {len(text)} characters of text from the document.")
        return text
    
    def chunk_text(self, text: str) -> List[str]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks.")
        return chunks
    
    def ensure_index_exists(self, index_name: str) -> None:

        if index_name not in self.pinecone.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {index_name} with dimension {self.dimension}")
            self.pinecone.create_index(
                name=index_name,
                dimension=self.dimension,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not self.pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)
    
    def embed_and_store_chunks(self, chunks: List[str], index_name: str) -> None:

        self.ensure_index_exists(index_name)
        index = self.pinecone.Index(index_name)
        
        logger.info(f"Embedding and storing chunks in batches of {self.batch_size}...")
        
        for i in range(0, len(chunks), self.batch_size):
            chunk_batch = chunks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}...")

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
    
    def process_and_store_document(self, url: str) -> Dict[str, Any]:

        try:
            logger.info("--- Starting PDF Processing ---")
            
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            index_name = f"hackrx-doc-{url_hash}"
            self.current_index_name = index_name
            
            file_path = self.download_document(url)
            
            try:
                text = self.extract_text_from_pdf(file_path)
                
                if not text:
                    return {
                        "success": False,
                        "message": "No text could be extracted from the PDF",
                        "chunks_processed": 0,
                        "index_name": index_name
                    }
                
                chunks = self.chunk_text(text)
                
                self.embed_and_store_chunks(chunks, index_name)
                
                logger.info("All document chunks embedded and stored in Pinecone using Gemini.")
                
                return {
                    "success": True,
                    "message": "Document processed and stored successfully",
                    "text_length": len(text),
                    "chunks_processed": len(chunks),
                    "index_name": index_name
                }
                
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info("Temporary file deleted.")
                    
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error processing document: {str(e)}",
                "chunks_processed": 0,
                "index_name": getattr(self, 'current_index_name', None)
            }
    
    def query_documents(self, query_text: str, top_k: int = 5, index_name: Optional[str] = None) -> List[Dict[str, Any]]:

        try:
            search_index_name = index_name or self.current_index_name
            
            if not search_index_name:
                logger.error("No index name available for querying")
                return []

            result = genai.embed_content(
                model="models/embedding-001",
                content=[query_text],
                task_type="RETRIEVAL_QUERY"
            )
            
            query_embedding = result['embedding'][0]

            index = self.pinecone.Index(search_index_name)
            search_results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            results = []
            for match in search_results['matches']:
                results.append({
                    "text": match['metadata']['text'],
                    "score": match['score'],
                    "id": match['id']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return []


def process_and_store_document(url: str) -> Dict[str, Any]:
    processor = DocumentProcessor()
    return processor.process_and_store_document(url)


if __name__ == "__main__":
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    processor = DocumentProcessor()
    result = processor.process_and_store_document(document_url)
    print(f"Processing result: {result}")