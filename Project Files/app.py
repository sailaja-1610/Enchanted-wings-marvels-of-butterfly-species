
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
import os
import requests
from bs4 import BeautifulSoup
import uuid

app = Flask(__name__)
model = load_model("best_model_mobilenet.keras")
label_encoder = joblib.load("label_encoder.pkl")

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            label = label_encoder.inverse_transform([class_index])[0]

            # Fetch Wikipedia info
            search_url = f"https://en.wikipedia.org/wiki/{label.replace(' ', '_')}"
            try:
                response = requests.get(search_url, timeout=5)
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = soup.select("p")
                description = next((p.text.strip() for p in paragraphs if len(p.text.strip()) > 60), "No description found.")
            except Exception:
                description = "No description found."

            return render_template("index.html", label=label, image=f"uploads/{filename}", description=description)

    return render_template("index.html", label=None)

if __name__ == "__main__":
    app.run(debug=True)
