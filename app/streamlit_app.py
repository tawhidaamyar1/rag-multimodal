import streamlit as st
from ingest import ingest
from agent import ask_agent
from clip_search import search_images

st.title("RAG Multimodal Agentique")

up = st.file_uploader("Charger un PDF", type="pdf")
if up and st.button("Indexer ce PDF"):
    path = "data/uploaded.pdf"
    with open(path, "wb") as f:
        f.write(up.read())
    with st.spinner("Indexation en cours (texte, tableaux, images)..."):
        n = ingest(path)
    st.success(f"{n} éléments indexés")

q = st.text_input("Pose ta question :")
if q:
    with st.spinner("Recherche..."):
        st.markdown(ask_agent(q))
    st.subheader("Figures proches (CLIP)")
    for p in search_images(q):
        st.image(p, width=400)