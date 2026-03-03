"""
main.py — API FastAPI pour le chatbot RAG (LangChain 1.0)

Démarrage :
    uvicorn src.main:app --reload

Exemples de requêtes :
    curl http://localhost:8000/health

    curl -X POST http://localhost:8000/chat \\
         -H 'Content-Type: application/json' \\
         -d '{"question": "De quoi parle le document ?", "session_id": "user-1"}'
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Corps de la requête POST /chat."""

    question: str
    session_id: str


class ChatResponse(BaseModel):
    """Corps de la réponse du endpoint /chat."""

    answer: str
    session_id: str


# ---------------------------------------------------------------------------
# Chargement de l'agent au démarrage
# ---------------------------------------------------------------------------

# L'agent est chargé une seule fois au démarrage pour éviter de reconstruire
# l'index FAISS à chaque requête.
rag_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement des ressources au démarrage, libération à l'arrêt."""
    global rag_agent
    print("[startup] Chargement de l'agent RAG...")
    from src.rag_engine import get_rag_agent
    rag_agent = get_rag_agent()
    print("[startup] Agent RAG prêt.")
    yield
    print("[shutdown] Arrêt de l'application.")


app = FastAPI(
    title="RAG Chatbot API",
    description="API REST pour un chatbot RAG construit avec LangChain 1.0 et Mistral AI.",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    """Vérifie que l'API est opérationnelle et l'agent chargé."""
    return {"status": "ok", "rag_agent_loaded": rag_agent is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint principal du chatbot RAG.

    Reçoit une question et un session_id, interroge l'agent RAG avec
    le thread correspondant et retourne la réponse.

    L'agent maintient automatiquement l'historique de conversation
    pour chaque session_id (= thread_id LangGraph).
    """
    if rag_agent is None:
        raise HTTPException(status_code=503, detail="L'agent RAG n'est pas encore chargé.")

    # TODO: Invoquez l'agent RAG avec la question et le session_id.
    #
    # L'agent attend :
    #   - un dict d'entrée  : {"messages": [{"role": "user", "content": request.question}]}
    #   - un config         : {"configurable": {"thread_id": request.session_id}}
    #
    # Il retourne un dict {"messages": [...]} dont le DERNIER élément
    # est l'AIMessage de réponse. Extrayez .content pour obtenir le texte.
    #
    # Exemple :
    #   result = rag_agent.invoke(
    #       {"messages": [{"role": "user", "content": ...}]},
    #       config={"configurable": {"thread_id": ...}},
    #   )
    #   answer = result["messages"][-1].content
    #
    # Remplacez les `...` par votre code :
    try:
        result = ...
        answer = ...
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'invocation de l'agent : {e}")

    return ChatResponse(answer=answer, session_id=request.session_id)
