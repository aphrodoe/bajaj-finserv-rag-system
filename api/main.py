from fastapi import FastAPI, status, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models import Query, Response
import secrets
import os
import dotenv

app = FastAPI()

security = HTTPBearer()

dotenv.load_dotenv()

API_KEY = os.getenv("API_KEY")

@app.post("/hackrx/run")
async def run_hackrx(query: Query, credentials: HTTPAuthorizationCredentials = Depends(security)):

    if not secrets.compare_digest(credentials.credentials, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    response = Response(answers=[])
    for question in query.questions:
        answer = question
        response.answers.append(answer)

    return response

