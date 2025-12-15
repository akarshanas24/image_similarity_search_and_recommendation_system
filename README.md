 AI Image Similarity Search & Recommendation System

An AI-powered system to find and recommend visually similar images using triplet-loss CNN embeddings and FAISS for fast nearest-neighbor search.

<img width="1920" height="1080" alt="Screenshot (326)" src="https://github.com/user-attachments/assets/e554a877-9726-4148-8e4b-e0b733843ea9" />


⚡ Features

Learn compact embeddings where similar images are close together using triplet-loss training.

Offline embedding generation and FAISS index for scalable similarity search.

Command-line interface to train, build index, and query images.

🛠️ Project Structure

train/                  # Training images
validation/             # Validation images
models/                 # Saved model checkpoints
Labels.json             # Image ID to label mapping
preprocess.py           # Data preprocessing
triplet_dataset.py      # Triplet dataset generator
train_triplet_model.py  # Training script
train_image_paths.pkl   # Serialized training image paths
train_index.faiss       # FAISS index
check_faiss_index.py    # Index verification
test_model.py           # Query similar images

⚡ Installation

git clone https://github.com/<your-username>/image_similarity_search_and_recommendation_system.git
cd image_similarity_search_and_recommendation_system

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt


Dependencies: torch, torchvision, faiss-cpu/faiss-gpu, numpy, Pillow, tqdm

🖼️ Usage

1️⃣ Prepare Data

Place images in train/ and validation/.

Update paths in preprocess.py and triplet_dataset.py.

python preprocess.py

2️⃣ Train Model
python train_triplet_model.py


Builds triplet datasets, trains CNN, saves best model in models/.

3️⃣ Build / Check FAISS Index
python check_faiss_index.py


Computes or verifies embeddings and index.

4️⃣ Query Similar Images
python test_model.py --query_image path/to/query.jpg --top_k 5


Returns top‑k similar images with labels.

🔍 How It Works
[Input Image] → [CNN Embedding] → [FAISS Nearest Neighbor Search] → [Top-K Similar Images]
