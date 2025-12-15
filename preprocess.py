import os
from collections import defaultdict

train_dir = 'train'
class_to_images = defaultdict(list)

# Loop over every sub-folder in 'train' directory
for category in os.listdir(train_dir):
    category_folder = os.path.join(train_dir, category)
    if os.path.isdir(category_folder):
        for img_file in os.listdir(category_folder):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(category_folder, img_file)
                class_to_images[category].append(img_path)

print("Loaded classes (folder-wise):")
for cls, imgs in class_to_images.items():
    print(f"{cls}: {len(imgs)} images")

# Sample preprocessing test
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def load_and_preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    return preprocess(img)

sample_class = list(class_to_images.keys())[0]
sample_img_path = class_to_images[sample_class][0]
tensor = load_and_preprocess(sample_img_path)
print(f"Sample tensor shape: {tensor.shape}")
