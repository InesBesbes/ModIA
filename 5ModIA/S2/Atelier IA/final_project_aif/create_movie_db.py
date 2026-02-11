## a été fait localement
import os
import torch
import annoy
import pickle
import gdown
import numpy as np
from PIL import Image
from torchvision import transforms, datasets, models

file_id = "1Pe46Rd62Alfu_m6rtOcX5ERN5ymBh6CI"
url = f"https://drive.google.com/uc?id={file_id}"
output = "mobilnet.pth"
gdown.download(url, output, quiet=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mobilenet = models.mobilenet_v3_small()
mobilenet.classifier[-1] = torch.nn.Linear(in_features=1024, out_features=10)
model = mobilenet.to(device)
model.load_state_dict(torch.load("./mobilnet.pth", map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((185, 185)),
    transforms.ToTensor(),
])

dataset_path = "../MovieGenre/content/sorted_movie_posters_paligema"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

vecteur_dim = 576
index = annoy.AnnoyIndex(vecteur_dim, 'angular')

movie_db = {}

print("Indexation des images...")

for idx, (path, _) in enumerate(dataset.imgs):
    image_pil = Image.open(path).convert("RGB")
    tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        vector = model.features(tensor)
        vector = torch.nn.functional.adaptive_avg_pool2d(vector, (1, 1))
        vector = torch.flatten(vector, start_dim=1)

    vector_np = vector.squeeze().cpu().numpy()
    index.add_item(idx, vector_np)

    movie_db[idx] = {
        "vector": vector_np,
        "image": np.array(image_pil, dtype=np.uint8)
    }

index.build(10)
index.save("movie_annoy_index.ann") ## on s'en sert pas au final

with open("movie_db.pkl", "wb") as f:
    pickle.dump(movie_db, f)

print("Index et base de données sauvegardés avec succès.")
