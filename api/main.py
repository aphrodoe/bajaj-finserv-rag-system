import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, status, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models import Query, Response
import secrets
import dotenv
import google.generativeai as genai

from document_ingestion.document import DocumentProcessor

app = FastAPI(title="HackRX API", description="API for document processing and querying")

security = HTTPBearer()

dotenv.load_dotenv()

API_KEY = os.getenv("API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

processor = DocumentProcessor()


def generate_answer_with_context(question: str, context_chunks: list) -> str:

    if not context_chunks:
        return "I couldn't find relevant information in the document to answer this question."
    
    context = "\n\n".join([chunk["text"] for chunk in context_chunks])
    
    prompt = f"""You are an expert Q&A system. Your task is to answer the user's question based *only* on the provided document context.
Analyze the context below and provide a concise, factual answer to the question.
If the answer cannot be found in the provided context, state "The provided document does not contain information to answer this question." Do not use any external knowledge.

[Document Context]
{context}
[/Document Context]

[Question]
{question}
[/Question]

Answer:"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


@app.post("/hackrx/run")
async def run_hackrx(query: Query, credentials: HTTPAuthorizationCredentials = Depends(security)):

    if not secrets.compare_digest(credentials.credentials, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        process_result = processor.process_and_store_document(query.documents)
        
        if not process_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process document: {process_result['message']}"
            )
        
        index_name = process_result.get("index_name")

        response = Response(answers=[])
        
        for question in query.questions:
            try:
                search_results = processor.query_documents(question, top_k=5, index_name=index_name)
                
                answer = generate_answer_with_context(question, search_results)
                response.answers.append(answer)
                
            except Exception as e:
                error_answer = f"Error processing question: {str(e)}"
                response.answers.append(error_answer)

        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)