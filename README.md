AI Image Similarity Search and Recommendation System

An AI-powered image similarity search and recommendation system that learns visual embeddings with a triplet-loss CNN and serves fast nearest-neighbor queries using a FAISS index.​​

Features
Triplet-loss training for learning a compact embedding space where similar images are close together.​​

Offline embedding generation and FAISS index for scalable similarity search.​

Command-line interface to train the model, build/check the index, and query similar images.

Project Structure
train/ – Training images.

validation/ – Validation images.

models/ – Saved model checkpoints and related artifacts.

Labels.json – Mapping from image paths / IDs to labels or human-readable names.

preprocess.py – Data preparation: reading images, resizing/normalizing, building label mappings.

triplet_dataset.py – Dataset utilities that generate triplets (anchor, positive, negative) for training.​​

train_triplet_model.py – Main training script for the triplet-loss network.

train_image_paths.pkl – Serialized list of training image paths used when building and querying the index.

train_index.faiss – FAISS index over image embeddings for fast similarity search.​

check_faiss_index.py – Utility to inspect / validate the FAISS index.

test_model.py – Script to embed query images and retrieve similar images from the index.

Installation
bash
git clone https://github.com/<your-username>/image_similarity_search_and_recommendation_system.git
cd image_similarity_search_and_recommendation_system

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
Typical dependencies: torch, torchvision, faiss-cpu or faiss-gpu, numpy, Pillow, tqdm.​

Data Preparation
Organize images:

Place training images in train/.

Place validation images in validation/.

Update any dataset paths / config variables in preprocess.py and triplet_dataset.py to match your folder names.

Run preprocessing:

bash
python preprocess.py
This step generates metadata such as Labels.json and train_image_paths.pkl required for training and indexing.

Training the Triplet Model
Train the embedding model:

bash
python train_triplet_model.py
This script will:

Build triplet datasets from train/ and validation/.

Initialize the CNN backbone (e.g., ResNet) and embedding head.

Optimize with triplet loss and save the best checkpoint into models/.​​

Adjust hyperparameters (batch size, learning rate, embedding dim, margin, epochs) in train_triplet_model.py.

Building and Checking the FAISS Index
If the index is not built automatically during training, generate or verify it using:

bash
python check_faiss_index.py
Expected behavior:

Load the best model from models/.

Compute or load stored embeddings for all images in train_image_paths.pkl.

Create / check train_index.faiss, printing example nearest neighbors for sanity checks.​

Querying Similar Images
Use test_model.py to retrieve similar images for a given query:

bash
python test_model.py \
  --query_image path/to/query.jpg \
  --top_k 5
The script will:

Load the trained model and train_index.faiss.

Embed the query image.

Return the top‑k most similar images along with their labels from Labels.json.​
