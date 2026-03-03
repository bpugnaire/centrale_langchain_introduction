"""
rag_engine.py — Moteur RAG réutilisable (LangChain 1.0)

Ce module expose `get_rag_agent()` qui construit et retourne un agent RAG
complet, prêt à être utilisé par l'API FastAPI.

Instructions pour les étudiants :
  1. Assurez-vous d'avoir sauvegardé l'index FAISS depuis le notebook
     (cellule vectorstore.save_local("../data/faiss_index")).
  2. Complétez les blocs TODO en vous appuyant sur votre Partie 3 du notebook.
  3. Testez ce module en isolation : `python -m src.rag_engine`
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index")
PDF_DIR = str(DATA_DIR)

RAG_SYSTEM_PROMPT = (
    "Tu es un assistant expert en analyse documentaire. "
    "Tu as UNIQUEMENT accès aux informations contenues dans les documents chargés dans le système. "
    "RÈGLE ABSOLUE : avant de répondre à n'importe quelle question, tu DOIS appeler l'outil "
    "`retrieve_context` avec une requête de recherche pertinente. "
    "Ne réponds jamais sans avoir d'abord consulté les documents via cet outil. "
    "Si le contexte récupéré ne contient pas la réponse, indique-le clairement sans inventer. "
    "Réponds toujours en français."
)

# ---------------------------------------------------------------------------
# Construction de l'agent RAG
# ---------------------------------------------------------------------------


def get_rag_agent():
    """
    Construit et retourne l'agent RAG complet.

    L'agent retourné attend en entrée :
        {"messages": [{"role": "user", "content": "..."}]}
    et doit être invoqué avec un config portant un thread_id :
        agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "..."}})

    La réponse est un dict {"messages": [...]} dont le dernier élément
    est l'AIMessage de réponse : result["messages"][-1].content

    Returns
    -------
    CompiledGraph
        L'agent RAG avec mémoire intégrée.
    """
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    # ------------------------------------------------------------------
    # 1. Modèle de langage et modèle d'embedding
    # ------------------------------------------------------------------
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
    embedding_model = MistralAIEmbeddings(model="mistral-embed")

    # ------------------------------------------------------------------
    # 2. Base vectorielle FAISS
    #    Chargement depuis le disque si disponible, reconstruction sinon.
    # ------------------------------------------------------------------
    faiss_index = Path(FAISS_INDEX_PATH)
    if faiss_index.exists():
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    else:
        print(
            f"[rag_engine] Index FAISS introuvable dans {FAISS_INDEX_PATH}. "
            "Reconstruction depuis les PDFs..."
        )
        loader = PyPDFDirectoryLoader(PDF_DIR)
        docs = loader.load()
        if not docs:
            raise FileNotFoundError(
                f"Aucun PDF trouvé dans {PDF_DIR}. "
                "Placez vos documents dans data/ avant de démarrer l'API."
            )
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embedding_model)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"[rag_engine] Index FAISS sauvegardé dans {FAISS_INDEX_PATH}.")

    # ------------------------------------------------------------------
    # 3. Tool de retrieval
    # ------------------------------------------------------------------

    # TODO: Copiez-collez ici votre tool `retrieve_context` de la Partie 3 du notebook.
    #
    # Rappel de la structure attendue :
    #
    #   @tool(response_format="content_and_artifact")
    #   def retrieve_context(query: str):
    #       """Retrieve information from documents to answer a query."""
    #       docs = vectorstore.similarity_search(query, k=3)
    #       serialized = "\n\n".join(
    #           f"Source: {doc.metadata}\nContent: {doc.page_content}"
    #           for doc in docs
    #       )
    #       return serialized, docs
    #
    # La variable `vectorstore` est disponible dans ce scope.

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        # TODO: implémentez le tool (voir Partie 3 du notebook ou rag_engine_solution.py)
        ...

    # ------------------------------------------------------------------
    # 4. Agent RAG
    # ------------------------------------------------------------------

    # TODO: Copiez-collez ici votre appel create_agent de la Partie 3 du notebook.
    #
    # Rappel de la structure attendue :
    #
    #   rag_agent = create_agent(
    #       model=llm,
    #       tools=[retrieve_context],
    #       system_prompt=RAG_SYSTEM_PROMPT,
    #       checkpointer=InMemorySaver(),
    #   )
    #
    # Les variables llm, retrieve_context et RAG_SYSTEM_PROMPT sont disponibles.

    rag_agent = ...

    return rag_agent


# ---------------------------------------------------------------------------
# Test rapide en ligne de commande
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = get_rag_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "De quoi traitent les documents chargés ?"}]},
        config={"configurable": {"thread_id": "test-cli"}},
    )
    print(result["messages"][-1].content)
