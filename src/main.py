"""
main.py — API FastAPI pour le chatbot RAG (LangChain 1.0)

FastAPI est un framework Python pour créer des APIs REST.
Il génère automatiquement une documentation interactive accessible sur /docs.

Démarrage :
    uvicorn src.main:app --reload

    uvicorn est le serveur web qui exécute l'application.
    --reload redémarre automatiquement le serveur à chaque modification du code.

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
#
# Pydantic valide automatiquement le JSON entrant et sortant.
# Si le client envoie un corps de requête incomplet ou mal typé,
# FastAPI renvoie une erreur 422 sans que vous ayez à écrire de validation.
#
# BaseModel transforme une classe Python en schéma JSON :
#   {"question": "...", "session_id": "..."} ↔ ChatRequest


class ChatRequest(BaseModel):
    """Corps de la requête POST /chat."""

    question: str      # La question posée par l'utilisateur
    session_id: str    # Identifiant de session — permet de maintenir la mémoire


class ChatResponse(BaseModel):
    """Corps de la réponse du endpoint /chat."""

    answer: str        # La réponse générée par l'agent
    session_id: str    # Renvoyé tel quel pour que le client puisse le suivre


# ---------------------------------------------------------------------------
# Chargement de l'agent au démarrage
# ---------------------------------------------------------------------------

# Variable globale : l'agent est construit une seule fois au démarrage.
# Cela évite de reconstruire l'index FAISS (coûteux) à chaque requête.
rag_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application.

    Le code avant `yield` s'exécute au démarrage du serveur (startup).
    Le code après `yield` s'exécute à l'arrêt (shutdown).

    C'est ici qu'on initialise les ressources lourdes : index vectoriel,
    connexions à des bases de données, chargement de modèles, etc.
    """
    global rag_agent
    print("[startup] Chargement de l'agent RAG...")
    from src.rag_engine import get_rag_agent
    rag_agent = get_rag_agent()
    print("[startup] Agent RAG prêt.")
    yield
    print("[shutdown] Arrêt de l'application.")


# Instanciation de l'application FastAPI.
# lifespan= branche le gestionnaire de cycle de vie défini ci-dessus.
app = FastAPI(
    title="RAG Chatbot API",
    description="API REST pour un chatbot RAG construit avec LangChain 1.0 et Mistral AI.",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
#
# Un endpoint est une URL associée à une fonction Python.
# Le décorateur @app.get / @app.post définit la méthode HTTP et le chemin.
#
#   GET  /health  → lecture simple, pas de corps de requête
#   POST /chat    → envoi de données (la question), retourne une réponse


@app.get("/health")
def health_check():
    """Vérifie que l'API est opérationnelle et l'agent chargé."""
    return {"status": "ok", "rag_agent_loaded": rag_agent is not None}


# response_model=ChatResponse indique à FastAPI le schéma de la réponse.
# Il filtre et valide automatiquement ce que la fonction retourne.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint principal du chatbot RAG.

    FastAPI désérialise automatiquement le JSON entrant vers un objet
    ChatRequest. Les champs sont accessibles via request.question,
    request.session_id, etc.

    L'agent maintient l'historique de conversation par session_id :
    deux appels avec le même session_id partagent la même mémoire.
    """
    # HTTPException interrompt immédiatement la requête et renvoie
    # le code HTTP et le message d'erreur au client.
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

    # FastAPI sérialise cet objet en JSON et l'envoie au client.
    return ChatResponse(answer=answer, session_id=request.session_id)
