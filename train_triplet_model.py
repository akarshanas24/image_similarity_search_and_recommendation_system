import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from triplet_dataset import get_triplet_dataloader
from preprocess import class_to_images, preprocess

class TripletNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        base.fc = nn.Identity()
        self.encoder = base
        self.embed = nn.Linear(2048, 128)

    def forward(self, x):
        x = self.encoder(x)
        x = self.embed(x)
        return F.normalize(x, p=2, dim=1)

def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()

def load_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TripletNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

def train(resume_epoch=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = TripletNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    dataloader = get_triplet_dataloader(class_to_images, batch_size=32, transform=preprocess)
    os.makedirs('models', exist_ok=True)
    num_epochs = 10

    # Resume from a checkpoint if requested
    start_epoch = 0
    if resume_epoch is not None:
        checkpoint_path = f'models/triplet_model_epoch_{resume_epoch}.pth'
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Resuming from epoch {resume_epoch}")
            start_epoch = resume_epoch
        else:
            print(f"Checkpoint {checkpoint_path} not found, training from scratch.")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            a_embed = model(anchor)
            p_embed = model(positive)
            n_embed = model(negative)
            loss = triplet_loss(a_embed, p_embed, n_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        checkpoint_path = f'models/triplet_model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    # For epoch 2 resume: train(resume_epoch=1)
    # For fresh training: train()
    train(resume_epoch=1)
