# Generate FAISS index from embeddings

import torch
import faiss

d = torch.load("afted.pt") # See Zenodo for this file
embs = d["embeddings"].numpy()

index = faiss.IndexFlatIP(128)
index.add(embs)
print(index.ntotal)

D, I = index.search(embs[:32], 100) # Example of searching

faiss.write_index(index, "afted.index")
