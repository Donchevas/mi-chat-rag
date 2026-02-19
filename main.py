from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.auth
import google.auth.transport.requests
import requests
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
PROJECT_ID = os.getenv("PROJECT_ID", "iagen-gcp-cwmi")
LOCATION = os.getenv("LOCATION", "us-west1")
RAG_CORPUS = os.getenv(
    "RAG_CORPUS",
    "projects/iagen-gcp-cwmi/locations/us-west1/ragCorpora/4532873024948404224"
)
MODEL = os.getenv("MODEL", "gemini-2.0-flash-001")


def get_access_token():
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token


class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        token = get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        contents = []
        for msg in req.history:
            contents.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": req.message}]
        })

        payload = {
            "contents": contents,
            "tools": [
                {
                    "retrieval": {
                        "vertexRagStore": {
                            "ragResources": [
                                {"ragCorpus": RAG_CORPUS}
                            ],
                            "similarityTopK": 5
                        }
                    }
                }
            ],
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "Eres un asistente útil que responde preguntas "
                            "basándose en la información de la base de conocimiento. "
                            "Si no encuentras la información en los documentos, "
                            "indícalo claramente. Responde siempre en el mismo idioma "
                            "en que te haga la pregunta el usuario."
                        )
                    }
                ]
            }
        }

        url = (
            f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/"
            f"{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/"
            f"{MODEL}:generateContent"
        )

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        candidate = data.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        answer = " ".join(p.get("text", "") for p in parts if "text" in p)

        if not answer:
            raise ValueError("Respuesta vacía del modelo")

        return {"answer": answer}

    except requests.exceptions.HTTPError as e:
        detail = e.response.text if e.response else str(e)
        raise HTTPException(status_code=502, detail=f"Error Vertex AI: {detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

---

## 📄 `requirements.txt`
```
fastapi==0.115.6
uvicorn[standard]==0.32.1
google-auth==2.37.0
requests==2.32.3
python-multipart==0.0.20