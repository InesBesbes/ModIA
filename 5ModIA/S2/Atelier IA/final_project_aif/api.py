import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import torchvision.models as models
import numpy as np
import annoy
import pickle
import io
import base64
from annoy import AnnoyIndex
from flask import Flask, send_file
from PIL import Image, UnidentifiedImageError
from lime import lime_image
import matplotlib.pyplot as plt
import traceback
from captum.attr import IntegratedGradients
from captum.attr import Occlusion
from captum.attr import visualization as viz
from skimage.segmentation import mark_boundaries

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)

MODEL_PATH = '/app/weights.pth'  
MOVIE_DB = '/app/movie_db.pkl'
VECTOR_DB = '/app/vector_db.pkl'

mobilenet = models.mobilenet_v3_small()
mobilenet.classifier[-1] = torch.nn.Linear(in_features=1024, out_features=10)
model = mobilenet.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((185, 185)),  
    transforms.ToTensor(),  
])

with open(MOVIE_DB, "rb") as f:
    movie_db = pickle.load(f)
with open(VECTOR_DB, "rb") as f:
    vector_db = pickle.load(f)

id_to_title = {idx: vector_db[idx]["title"] for idx in vector_db}
title_to_id = {title: idx for idx, title in id_to_title.items()}

indexes = {}
for method in ["distilbert", "glove", "bow", "tfidf"]:
    dim = len(vector_db[0][method]) 
    index = AnnoyIndex(dim, 'angular')
    index.load(f"/app/{method}_index.ann") 
    indexes[method] = index

# ************************ Prédiction du genre du film ***********************************
@app.route('/predict', methods=['POST'])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary)).convert("RGB")
    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)

    return jsonify({"prediction": int(predicted[0])})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    images_binary = request.files.getlist("images[]")
    tensors = []

    for img_binary in images_binary:
        img_pil = Image.open(img_binary.stream).convert("RGB")
        tensor = transform(img_pil)
        tensors.append(tensor)

    batch_tensor = torch.stack(tensors, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predictions = outputs.max(1)

    return jsonify({"predictions": predictions.tolist()})


# ********************** Recommandation (image) *********************************

def extract_features(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        vector = model.features(tensor)
        vector = torch.nn.functional.adaptive_avg_pool2d(vector, (1, 1))
        vector = torch.flatten(vector, start_dim=1)

    return vector.squeeze().cpu().numpy()

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

@app.route("/recommend/", methods=["POST"])
def recommend():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    query_image_bytes = file.read()
    query_vector = extract_features(query_image_bytes)

    similarities = []
    for idx, entry in movie_db.items():
        db_vector = entry["vector"]
        sim = cosine_similarity(query_vector, db_vector)
        similarities.append((idx, sim))

    top_indices = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

    results = []
    for idx, sim in top_indices:
        img_array = movie_db[idx]["image"]
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        results.append(f"data:image/png;base64,{img_base64}")

    return jsonify({"movies": results})


# ********************** Recommandation (titre) *********************************

@app.route("/recommend_title", methods=["POST"])
def recommend_title():
    data = request.json
    title = data.get("title")
    method = data.get("method")

    if title not in title_to_id:
        return jsonify({"error": f"titre '{title}' pas trouvé lol"}), 400
    if method not in indexes:
        return jsonify({"error": f"méthode '{method}' pas supportée"}), 400

    idx = title_to_id[title]
    neighbors = indexes[method].get_nns_by_item(idx, 6)[1:] 
    recommended_titles = [id_to_title[n] for n in neighbors]

    return jsonify({"recommendations": recommended_titles})

# ********************** XAI *********************************
@app.route('/explain', methods=['POST'])
def explain():
    try:
        img_binary = request.data
        img_pil = Image.open(io.BytesIO(img_binary))

        tensor = transform(img_pil).to(device)
        tensor = tensor.unsqueeze(0)


        ####### Predicted label #########
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = outputs.max(1)

        def batch_predict(images):
            model.eval()
            batch = torch.stack([transform(Image.fromarray(img)) for img in images]).to(device)
            with torch.no_grad():
                logits = model(batch)
            return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        
        ####### LIME Explanation #########
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(img_pil), 
                                                batch_predict, top_labels=5,
                                                hide_color=0, 
                                                num_samples=1000) 
        
        lime_img, lime_mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, hide_rest=False)
        img_boundry = mark_boundaries(lime_img/255.0, lime_mask)

        fig_lime = plt.figure()
        plt.imshow(img_boundry, cmap='hot')
        plt.axis('off')

        lime_buf = io.BytesIO()
        fig_lime.savefig(lime_buf, format="png", bbox_inches='tight')
        plt.close(fig_lime)
        lime_buf.seek(0)
        lime_b64 = base64.b64encode(lime_buf.getvalue()).decode()
        
        ####### CAPTUM Occlusion #########
        occlusion = Occlusion(model)
        attributions_occl = occlusion.attribute(tensor,
                                        strides = (3, 8, 8),
                                        target=int(predicted[0]), 
                                        sliding_window_shapes=(3,15, 15),
                                        baselines=0)
        
        fig_occl, _ = viz.visualize_image_attr_multiple(
            attributions_occl.squeeze().cpu().detach().numpy().transpose(1, 2, 0),
            tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0),
            methods=["heat_map"],
            signs=["positive"],
            cmap='hot',
            show_colorbar=True
        )

        buf_occl = io.BytesIO()
        fig_occl.savefig(buf_occl, format="png", bbox_inches="tight")
        plt.close(fig_occl)
        buf_occl.seek(0)
        occl_b64 = base64.b64encode(buf_occl.getvalue()).decode()

        ####### CAPTUM Integrated Gradients #########
        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(tensor, 
                                                         target=int(predicted[0]), 
                                                         n_steps=200, 
                                                         internal_batch_size=1)

        fig_ig, _ = viz.visualize_image_attr_multiple(
            attributions_ig.squeeze().cpu().detach().numpy().transpose(1, 2, 0),
            tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0),
            methods=["heat_map"],
            signs=["positive"],
            cmap='hot',
            show_colorbar=True
        )

        buf = io.BytesIO()
        fig_ig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig_ig)
        buf.seek(0)
        ig_b64 = base64.b64encode(buf.getvalue()).decode()


        return jsonify({
            "lime": lime_b64,
            "occlusion": occl_b64,
            "integrated_gradients": ig_b64

        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ********************** Georges *********************************

IMAGE_PATH = '/app/photo.jpg'

@app.route("/get-image", methods=["GET"])
def get_image():
    try:
        print(f"Tentative de chargement de : {IMAGE_PATH}")
        image = Image.open(IMAGE_PATH).convert("RGB")
        img_io = io.BytesIO()
        image.save(img_io, format='JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    except UnidentifiedImageError as e:
        print("Erreur PIL : image non reconnue :", e)
        return f"Erreur : image non reconnue - {e}", 500
    except Exception as e:
        print("Erreur générale :", e)
        return f"Erreur serveur : {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)