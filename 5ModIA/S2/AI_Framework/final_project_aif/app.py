import gradio as gr
import requests
from PIL import Image
import io
import base64
import os


genre_dict = {
    0: "action",
    1: "animation",
    2: "comedy",
    3: "documentary",
    4: "drama",
    5: "fantasy",
    6: "horror",
    7: "romance",
    8: "science fiction",
    9: "thriller"
}

API_URL = "http://model_api:5000"


# --- Onglet 1 : Prédiction de genre ---

def predict_genre(image):
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    img_binary.seek(0)

    try:
        response = requests.post(f"{API_URL}/predict", data=img_binary.getvalue())
        response.raise_for_status()
        prediction = response.json()["prediction"]
        return genre_dict.get(prediction, "Inconnu")
    except Exception as e:
        print("Erreur lors de la prédiction :", e)
        return "Erreur"

predictor_interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Image(type="pil", label="Image du film"),
    outputs=gr.Textbox(label="Genre Prédit"),
    title="Prédiction de Genre",
    description="Uploadez une affiche pour prédire son genre."
)

# --- Onglet 2 : Recommandation par image ---

def recommend_movies(image):
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    img_binary.seek(0)

    files = {"file": ("image.png", img_binary, "image/png")}
    
    response = requests.post(f"{API_URL}/recommend/", files=files)
    response.raise_for_status()
    similar_movies = response.json()["movies"]

    images = []
    for img_base64 in similar_movies:
        try:
            header, encoded = img_base64.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
        except Exception as e:
            print("Erreur lors du décodage de l'image :", e)
            images.append(None)

    return images

METHODS = ["glove", "distilbert", "bow", "tfidf"]

recommender_interface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Image(type="pil", label="Image du film"),
    outputs=[gr.Image(type="pil", label=f"Similar Movie {i+1}") for i in range(5)],
    title="Recommandation (image)",
    description="Uploadez une affiche de film pour obtenir des suggestions visuellement similaires."
)

# --- Onglet 3 : Recommandation par titre ---
def recommend_by_title(title, method):
    try:
        response = requests.post(f"{API_URL}/recommend_title", json={"title": title, "method": method})
        if response.status_code != 200:
            return f"Erreur : {response.json().get('error', 'inconnue')}"
        recos = response.json()["recommendations"]
        return f"\n".join([f"{i+1}. {t}" for i, t in enumerate(recos)])
    except Exception as e:
        return f"Erreur de connexion : {e}"

title_reco_interface = gr.Interface(
    fn=recommend_by_title,
    inputs=[
        gr.Textbox(label="Titre du Film", placeholder="The Matrix"),
        gr.Dropdown(METHODS, label="Méthode d'embedding", value="glove")
    ],
    outputs=gr.Textbox(label="Films Recommandés"),
    title="Recommandation (titre)",
    description="Entrez le nom d'un film et choisissez la méthode d'embedding pour voir les films les plus similaires."
)
# --- Onglet 5 : XAI ---

def explain_image(image):
    # Convert image to bytes
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    
    # Send request to the API
    response = requests.post(f"{API_URL}/explain", data=img_binary.getvalue())
    data = response.json()
    lime_img = Image.open(io.BytesIO(base64.b64decode(data["lime"])))
    occl_img = Image.open(io.BytesIO(base64.b64decode(data["occlusion"])))
    ig_img = Image.open(io.BytesIO(base64.b64decode(data["integrated_gradients"])))
    
    return lime_img, ig_img, occl_img

explain_image_interface = gr.Interface(
    fn=explain_image,
    inputs=gr.Image(type="pil", label="Image à expliquer"),
    outputs=[
        gr.Image(type="pil", label="LIME Explanation"),
        gr.Image(type="pil", label="Integrated Gradients"),
        gr.Image(type="pil", label="Occlusion")
    ],
    title="Explications XAI",
    description="Uploadez une image pour obtenir des explications sur la prédiction du modèle."
)

# --- Onglet 5 : Georges ---

def get_image_from_api():
    try:
        response = requests.get(f"{API_URL}/get-image") 
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image_path = "/tmp/image_from_drive.png"
        image.save(image_path)
        return image_path
    except Exception as e:
        print("Erreur lors de la récupération de l'image :", e)
        return None

image_interface = gr.Interface(
    fn=get_image_from_api,
    inputs=[],
    outputs=gr.Image(type="filepath", label="le plus beau chat du monde"),
    title="Georges"
)

gr.TabbedInterface(
    interface_list=[predictor_interface, recommender_interface, title_reco_interface, explain_image_interface, image_interface],
    tab_names=["Prédiction de Genre", "Recommandation (image)", "Recommandation (titre)", "XAI", "Georges"]
).launch(server_name="0.0.0.0", server_port=7860)

