"""
app.py — Interface Chainlit pour le chatbot RAG (LangChain 1.0)

Démarrage :
    chainlit run app.py

L'agent est partagé entre toutes les sessions ; chaque session dispose
de sa propre mémoire grâce au thread_id = identifiant de session Chainlit.
"""

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

load_dotenv()

# Chargement unique de l'agent au démarrage du processus
from src.rag_engine import get_rag_agent

_agent = get_rag_agent()


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="Bonjour ! Je suis votre assistant documentaire. Posez-moi une question sur les documents chargés."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.context.session.id

    result = await _agent.ainvoke(
        {"messages": [{"role": "user", "content": message.content}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    messages = result["messages"]

    # Affichage des appels de tools sous forme de Steps cliquables
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                query = call["args"].get("query", "")
                # Récupération de la réponse du tool (ToolMessage suivant)
                tool_output = ""
                for j in range(i + 1, len(messages)):
                    if isinstance(messages[j], ToolMessage):
                        tool_output = messages[j].content[:500]
                        break
                async with cl.Step(name=f"Recherche : {query}", type="tool") as step:
                    step.output = tool_output

    await cl.Message(content=messages[-1].content).send()
