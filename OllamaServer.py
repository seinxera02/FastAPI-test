from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import uvicorn

app = FastAPI()

# Allow external access (important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

class Question(BaseModel):
    message: str

@app.post("/ask")
async def ask_question(question: Question):
    ollama_payload = {
        "model": "qwen2.5:1.5b",
        "prompt": "Sujal is a boy who is currently doing Bachelors in Computer Science."+question.message,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=ollama_payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        return {
            "answer": data.get("response", "")
        }

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running on the host machine."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",  # VERY IMPORTANT
        port=8000
    )
