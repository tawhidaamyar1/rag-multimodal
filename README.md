# RAG Multimodal Agentique - Étape 1

Question-réponse sur un PDF avec texte, tableaux et images (« Attention Is All You Need »).

## Architecture
PDF → PyMuPDF/pdfplumber (texte, tableaux, images) → embeddings BGE-M3 → PGVector.
Images → description LLaVA (indexée) + recherche CLIP.
Question → recherche texte (PGVector) → agent LangChain (Qwen2.5 via Ollama) → réponse avec pages → Streamlit.

## Modèles et outils
- Embeddings : BAAI/bge-m3
- LLM : qwen2.5:3b (Ollama)
- Description d'images : llava:7b (Ollama)
- Images : CLIP (clip-ViT-B-32)
- Base : PostgreSQL + pgvector (Docker, port 5433)
- Frameworks : LlamaIndex, LangChain, Streamlit

## Lancement
```
docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragdb -p 5433:5432 pgvector/pgvector:pg16
docker exec -it pgvector psql -U postgres -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
pip install -r requirements.txt
python app/ingest.py
streamlit run app/streamlit_app.py
```

## Exemples
- « Combien de couches contient l'encodeur du Transformer ? » → 6 couches
- « Quel est le score BLEU du Transformer big en EN-DE ? » → 28.4
- « Montre-moi l'architecture du Transformer » → réponse + figures

## Limites
- Modèle 3b : réponses parfois imprécises.
- CLIP comprend mieux les requêtes en anglais.
- Avec un modèle 3b, les outils sont sélectionnés par règles (recherche texte toujours, figures si la question les demande) ; l'agent LangChain génère la réponse.
- Réindexer un même PDF crée des doublons.
