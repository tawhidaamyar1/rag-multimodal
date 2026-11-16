import pymupdf, pdfplumber, ollama
from llama_index.core import Settings, VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

PDF = "data/attention_is_all_you_need.pdf"
VISION_MODEL = "llava:7b"   # mets "qwen2.5vl:7b" si tu l'as téléchargé

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def describe_image(path):
    r = ollama.chat(model=VISION_MODEL, messages=[{
        "role": "user",
        "content": "Décris cette figure techniquement en français.",
        "images": [path]}])
    return r["message"]["content"]


docs = []
pdf = pymupdf.open(PDF)
with pdfplumber.open(PDF) as plumber:
    for i, page in enumerate(pdf, start=1):
        # TEXTE
        docs.append(Document(text=page.get_text(),
                             metadata={"type": "text", "page": i, "source": PDF}))
        # TABLEAUX
        for t in plumber.pages[i - 1].extract_tables():
            md = "\n".join(" | ".join(str(c) for c in row) for row in t)
            docs.append(Document(text=md,
                                 metadata={"type": "table", "page": i, "source": PDF}))
        # IMAGES -> fichier + description textuelle
        for j, img in enumerate(page.get_images(full=True)):
            pix = pymupdf.Pixmap(pdf, img[0])
            if pix.n > 4:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            path = f"images/p{i}_{j}.png"
            pix.save(path)
            docs.append(Document(text=describe_image(path),
                                 metadata={"type": "image", "page": i,
                                           "image_path": path, "source": PDF}))

vector_store = PGVectorStore.from_params(
       host="localhost", port=5433, user="postgres", password="postgres",
    database="ragdb", table_name="rag_docs", embed_dim=1024)
VectorStoreIndex.from_documents(
    docs, storage_context=StorageContext.from_defaults(vector_store=vector_store))
print("Indexation terminée :", len(docs), "éléments")