import glob
from PIL import Image
from sentence_transformers import SentenceTransformer, util

clip = SentenceTransformer("clip-ViT-B-32")
paths = sorted(glob.glob("images/*.png"))
img_emb = clip.encode([Image.open(p) for p in paths], convert_to_tensor=True)


def search_images(query, k=2):
    q = clip.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q, img_emb, top_k=k)[0]
    return [paths[h["corpus_id"]] for h in hits]


if __name__ == "__main__":
    print(search_images("transformer architecture encoder decoder"))