import os
import random
import torch
from PIL import Image
from train_triplet_model import load_model
from preprocess import preprocess
import matplotlib.pyplot as plt
import numpy as np
import faiss
import pickle

train_folder = 'train'
test_folder = 'validation'
checkpoint_path = 'models/triplet_model_epoch_1.pth'
K = 5

index_path = 'train_index.faiss'
img_map_path = 'train_image_paths.pkl'

# ======================== DB BUILD: ENCODE AND SAVE ==========================
# Only run this block if index/img_map do not exist OR if you update train set!
if not (os.path.exists(index_path) and os.path.exists(img_map_path)):
    print("(Re)Building DB: Encoding and indexing all train images...")

    # Model & device
    model = load_model(checkpoint_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Gather paths
    train_image_paths = []
    for class_folder in os.listdir(train_folder):
        class_path = os.path.join(train_folder, class_folder)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    train_image_paths.append(os.path.join(class_path, file))

    print(f"Found {len(train_image_paths)} train images.")

    # Embeddings
    embeddings = []
    with torch.no_grad():
        for img_path in train_image_paths:
            img = Image.open(img_path).convert('RGB')
            tensor = preprocess(img).unsqueeze(0).to(device)
            embedding = model(tensor)
            embeddings.append(embedding.cpu().numpy())
    embedding_matrix = np.vstack(embeddings).astype('float32')
    faiss.normalize_L2(embedding_matrix)
    dim = embedding_matrix.shape[1]

    # Index and map
    index = faiss.IndexFlatIP(dim)
    index.add(embedding_matrix)
    faiss.write_index(index, index_path)
    with open(img_map_path, 'wb') as f:
        pickle.dump(train_image_paths, f)
    print(f"FAISS index and image map saved ({index_path}, {img_map_path})")

else:
    print("Loading existing DB")

# ======================== QUERY & VISUALIZATION ==============================

# Load index and image mapping
index = faiss.read_index(index_path)
with open(img_map_path, 'rb') as f:
    train_image_paths = pickle.load(f)

# Load test image paths
test_image_paths = []
for class_folder in os.listdir(test_folder):
    class_path = os.path.join(test_folder, class_folder)
    if os.path.isdir(class_path):
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_image_paths.append(os.path.join(class_path, file))

# Pick a random test image
query_image_path = random.choice(test_image_paths)
query_img = Image.open(query_image_path).convert('RGB')

# Load model for query embedding
model = load_model(checkpoint_path)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
query_tensor = preprocess(query_img).unsqueeze(0).to(device)
with torch.no_grad():
    q_emb = model(query_tensor).cpu().numpy().astype('float32')
faiss.normalize_L2(q_emb)

D, I = index.search(q_emb, k=K)
top_paths = [train_image_paths[i] for i in I[0]]
top_scores = D[0]

print(f"\nQuery Image: {query_image_path}\n")
print("🔍 Top 5 Most Similar Images (cosine similarity):")
for r, (img_path, score) in enumerate(zip(top_paths, top_scores), 1):
    print(f"{r}. {img_path} — Cosine Similarity: {score:.4f}")

# Visualization
fig = plt.figure(figsize=(18, 6))
fig.suptitle('Image Similarity Search Results', fontsize=16, fontweight='bold')
ax = plt.subplot(2, 3, 1)
ax.imshow(query_img)
ax.set_title('Query Image', fontsize=14, fontweight='bold', color='blue')
ax.axis('off')
for spine in ax.spines.values():
    spine.set_edgecolor('blue')
    spine.set_linewidth(3)
    spine.set_visible(True)
for i in range(K):
    img = Image.open(top_paths[i]).convert('RGB')
    ax = plt.subplot(2, 3, i + 2)
    ax.imshow(img)
    ax.set_title(f'Rank {i+1}\nSim: {top_scores[i]:.4f}', fontsize=12)
    ax.axis('off')
    color = 'green' if top_scores[i] > 0.85 else 'orange' if top_scores[i] > 0.70 else 'red'
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
        spine.set_visible(True)
plt.tight_layout()
plt.show()
