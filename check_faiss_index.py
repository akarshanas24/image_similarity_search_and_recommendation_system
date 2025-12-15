import faiss

index = faiss.read_index('train_index.faiss')
print("Number of stored vectors:", index.ntotal)
print("Vector dimension:", index.d)
