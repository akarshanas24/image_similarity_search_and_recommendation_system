AI Image Similarity Search and Recommendation System

An AI-powered system to find and recommend visually similar images using a triplet-loss CNN and FAISS for fast nearest-neighbor search.

Features

Triplet-loss training to map similar images close in embedding space.

Offline embedding generation and FAISS index for scalable similarity search.

Command-line interface for training, indexing, and querying images.

Project Structure
train/                  # Training images
validation/             # Validation images
models/                 # Saved model checkpoints
Labels.json             # Image ID to label mapping
preprocess.py           # Data preparation & preprocessing
triplet_dataset.py      # Triplet dataset generator
train_triplet_model.py  # Training script
train_image_paths.pkl   # Serialized training image paths
train_index.faiss       # FAISS index for similarity search
check_faiss_index.py    # Index verification utility
test_model.py           # Query similar images

Installation
git clone https://github.com/<your-username>/image_similarity_search_and_recommendation_system.git
cd image_similarity_search_and_recommendation_system
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt


Dependencies: torch, torchvision, faiss-cpu/faiss-gpu, numpy, Pillow, tqdm

Usage

1. Prepare Data

Place images in train/ and validation/.

Update paths in preprocess.py and triplet_dataset.py.

python preprocess.py


2. Train Model

python train_triplet_model.py


Trains CNN with triplet loss.

Saves best model to models/.

3. Build / Check FAISS Index

python check_faiss_index.py


4. Query Similar Images

python test_model.py --query_image path/to/query.jpg --top_k 5


Returns top‑k similar images with labels.
