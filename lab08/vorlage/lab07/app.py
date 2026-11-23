# lab07/app.py - der Dienst des Traegermodells
# Starten (aus dem Hauptordner des Repos):  uvicorn lab07.app:app
import base64, io, os

import mlflow.pytorch
import yaml
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from lab02.src.traeger import make_transform, predict     # dieselbe Vorverarbeitung wie im Training - importiert, nicht abgetippt

# ---- laeuft EINMAL beim Start -------------------------------------------------------------------------------
MODEL_URI = os.getenv("MODEL_URI", "models:/traeger@champion")     # Umgebungsvariable MODEL_URI, sonst der Alias
model = mlflow.pytorch.load_model(MODEL_URI).eval()
tf = make_transform(train=False)
classes = [c["name"] for c in yaml.safe_load(open("models/traeger_v1.yaml"))["classes"]]


# ---- Schema: was hinein darf und was herauskommt ----------------------------------------------------------------
class Anfrage(BaseModel):
    image_b64: str                 # das Bild als Base64-Text


class Antwort(BaseModel):
    klasse: str                    # Name der vorhergesagten Klasse
    p: float                       # ihre Wahrscheinlichkeit (Softmax)
    modell: str                    # welches Modell geantwortet hat


app = FastAPI(title="Trägermodell-Service")


# ---- Endpunkte ---------------------------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "modell": MODEL_URI}           # Lebenszeichen - benutzt das Modell nicht


@app.post("/predict", response_model=Antwort)
def vorhersage(a: Anfrage):                                # FastAPI prueft den JSON-Koerper gegen das Schema Anfrage
    try:
        bild = Image.open(io.BytesIO(base64.b64decode(a.image_b64))).convert("RGB")
    except Exception:
        raise HTTPException(422, "image_b64 ist kein lesbares Bild")
    name, p = predict(model, tf, classes, bild)            # Klassenname und Liste der 5 Wahrscheinlichkeiten
    return Antwort(klasse=name, p=round(max(p), 4), modell=MODEL_URI)
