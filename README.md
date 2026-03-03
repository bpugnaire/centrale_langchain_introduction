# TP LangChain 1.0 — Construire un agent RAG avec Mistral AI

TP destiné aux étudiants de dernière année d'école d'ingénieurs.

**Objectif :** maîtriser les fondamentaux de LangChain 1.0 — modèles, agents, mémoire, tools —
puis construire un chatbot RAG et l'exposer via une API REST FastAPI.

---

## Structure du projet

```
centrale_langchain_introduction/
├── data/                          # Vos fichiers PDF (à ajouter manuellement)
│   └── faiss_index/               # Index FAISS généré par le notebook
├── notebooks/
│   └── tp_langchain_exploration.ipynb   # Phase 1 — exploration guidée
├── src/
│   ├── __init__.py
│   ├── rag_engine.py              # Phase 2 — moteur RAG (get_rag_agent)
│   └── main.py                    # Phase 2 — API FastAPI
├── app.py                         # Phase 3 — interface Chainlit
├── .env.example                   # Template de configuration
├── .python-version                # Version Python utilisée par uv (3.13)
├── pyproject.toml                 # Dépendances du projet
└── README.md
```

---

## Prérequis

- Une clé API Mistral AI (https://console.mistral.ai/)
- [`uv`](https://docs.astral.sh/uv/) pour la gestion des dépendances

---

## Installation

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd centrale_langchain_introduction
```

### 2. Installer uv

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Installer Python 3.13 et les dépendances

**macOS / Linux**
```bash
uv sync
source .venv/bin/activate
```

**Windows** — `uv sync` peut échouer sur le téléchargement automatique de Python.
Installez d'abord Python 3.13 explicitement :
```powershell
uv python install 3.13
uv sync
.venv\Scripts\activate
```

### 4. Configurer la clé API

```bash
cp .env.example .env   # Linux / macOS
# copy .env.example .env   # Windows
```

Éditez `.env` et renseignez votre `MISTRAL_API_KEY`.

---

## Déroulement du TP

### Phase 1 — Exploration (notebook)

1. Placez un ou plusieurs fichiers PDF dans le dossier `data/`
2. Lancez Jupyter :
   ```bash
   jupyter notebook notebooks/tp_langchain_exploration.ipynb
   ```
3. Complétez les blocs `# TODO` dans les trois parties :

   | Partie | Concepts clés |
   |---|---|
   | **1 — Modèle & Prompts** | `ChatMistralAI`, `.invoke()`, `SystemMessage`, `HumanMessage` |
   | **2 — Agent & Mémoire** | `create_agent`, `InMemorySaver`, `thread_id`, `.stream()` |
   | **3 — RAG** | `FAISS`, `MistralAIEmbeddings`, `@tool`, `create_agent` avec retrieval |

> À la fin de la Partie 3, l'index FAISS est sauvegardé dans `data/faiss_index/`.

### Phase 2 — Industrialisation (API FastAPI)

1. Complétez `src/rag_engine.py` — copiez le `@tool` et le `create_agent` de la Partie 3
2. Complétez `src/main.py` — implémentez l'invocation dans l'endpoint `/chat`
3. Démarrez l'API :
   ```bash
   uvicorn src.main:app --reload
   ```
4. Testez :
   ```bash
   # Vérification
   curl http://localhost:8000/health

   # Question au chatbot
   curl -X POST http://localhost:8000/chat \
        -H 'Content-Type: application/json' \
        -d '{"question": "De quoi parle le document ?", "session_id": "test-1"}'

   # Deuxième question (même session) — teste la mémoire
   curl -X POST http://localhost:8000/chat \
        -H 'Content-Type: application/json' \
        -d '{"question": "Peux-tu développer ?", "session_id": "test-1"}'
   ```
5. Documentation interactive : http://localhost:8000/docs

### Phase 3 — Interface Chainlit (optionnel)

```bash
chainlit run app.py
```

Ouvre `http://localhost:8000` dans votre navigateur. Chaque onglet / session
dispose de sa propre mémoire de conversation.

L'interface affiche :
- les étapes de raisonnement de l'agent (quel outil a été appelé, avec quelle requête)
- les tokens streamés en temps réel
- l'historique de la conversation

---

## Stack technique (LangChain 1.0)

| Composant | Technologie |
|---|---|
| LLM | `ChatMistralAI` via `langchain-mistralai` |
| Initialisation de l'agent | `create_agent` (`langchain.agents`) |
| Tools | `@tool` decorator (`langchain.tools`) |
| Mémoire | `InMemorySaver` + `thread_id` (`langgraph`) |
| Embeddings | `MistralAIEmbeddings` (modèle `mistral-embed`) |
| Base vectorielle | FAISS (`faiss-cpu`) |
| Chargement PDF | `PyPDFDirectoryLoader` (`pypdf`) |
| API | FastAPI + Uvicorn |
| Interface chat | Chainlit |
| Gestion des secrets | `python-dotenv` |
| Gestion des dépendances | `uv` + `pyproject.toml` |
