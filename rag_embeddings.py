from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index(chunks):
    if not chunks:
        return faiss.IndexFlatIP(384), [], chunks  # IP = Inner Product

    embeddings = model.encode(chunks, normalize_embeddings=True)  # normalize!
    index = faiss.IndexFlatIP(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))

    return index, embeddings, chunks


def search(query, index, chunks, k=3):
    if not chunks or index.ntotal == 0:
        return []

    query_vec = model.encode([query], normalize_embeddings=True)  # normalize!
    query_vec = np.array(query_vec).astype('float32')

    scores, indices = index.search(query_vec, k)

    results = []
    for i, score in zip(indices[0], scores[0]):
        if i != -1 and i < len(chunks) and score > 0.3:  # cosine > 0.3, higher = more similar
            results.append(chunks[i])

    return results