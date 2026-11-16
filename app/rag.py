from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=300)

store = PGVectorStore.from_params(
    host="localhost", port=5433, user="postgres", password="postgres",
    database="ragdb", table_name="rag_docs", embed_dim=1024)
index = VectorStoreIndex.from_vector_store(store)
retriever = index.as_retriever(similarity_top_k=5)


def answer(question):
    nodes = retriever.retrieve(question)
    ctx = "\n\n".join(
        f"[page {n.metadata['page']} - {n.metadata['type']}]\n{n.text}"
        for n in nodes)
    prompt = f"""Réponds uniquement avec le contexte. Cite les pages.
Si la réponse n'est pas dans le contexte, réponds "Je ne sais pas".
<context>{ctx}</context>
<question>{question}</question>"""
    return str(Settings.llm.complete(prompt)), nodes


if __name__ == "__main__":
    rep, _ = answer("Combien de couches contient l'encodeur du Transformer ?")
    print(rep)