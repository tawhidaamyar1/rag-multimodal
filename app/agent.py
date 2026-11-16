from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from rag import retriever
from clip_search import search_images

FIGURE_WORDS = ["figure", "schéma", "schema", "architecture", "image",
                "diagramme", "montre", "show", "diagram"]


def search_text_tables(query):
    """Outil 1 : recherche texte, tableaux et descriptions d'images (PGVector)."""
    return "\n\n".join(
        f"[page {n.metadata['page']} | {n.metadata['type']}] {n.text}"
        for n in retriever.retrieve(query))


def search_figures(query):
    """Outil 2 : recherche de figures par similarité CLIP."""
    return search_images(query)


prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant documentaire. Réponds en français, en une "
               "ou deux phrases, uniquement avec le contexte, et cite la page. "
               "Réponds en texte simple, sans JSON."),
    ("user", "Contexte :\n{context}\n\nQuestion : {question}")])

chain = prompt | ChatOllama(model="qwen2.5:3b", temperature=0) | StrOutputParser()


def ask_agent(question):
    # Sélection d'outils : texte toujours, figures si la question les demande
    ctx = search_text_tables(question)
    answer = chain.invoke({"context": ctx, "question": question}).strip()
    if not answer:
        answer = "Aucune réponse générée, reformule ta question."
    if any(w in question.lower() for w in FIGURE_WORDS):
        answer += "\n\nFigures pertinentes : " + ", ".join(search_figures(question))
    return answer


if __name__ == "__main__":
    print(ask_agent("Combien de couches contient l'encodeur du Transformer ?"))