"""Stufe 4 von 4 der Traegermodell-Pipeline (AKE Lab 6): evaluate.

Laedt das Traegermodell NUR aus models/traeger_v1.yaml + traeger_v1.pt (mit load_traeger, wie spaeter der
Service) und bewertet es auf den Validierungsbildern des Manifests (Lab 2, Aufgabe 5).
Ausgabe: lab06/reports/metrics.json - eine kleine Datei, die in Git bleibt (dvc metrics show / diff).

Aufruf (aus dem Hauptordner des Uebungs-Repos):  python lab06/evaluate.py
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, "lab02")
from src.traeger import load_traeger

torch.set_num_threads(min(4, os.cpu_count()))
t0 = time.perf_counter()
ROOT = Path("data/imagenette5")
MANIFEST = Path("lab06/data/manifest.csv")

model, tf, classes = load_traeger("models/traeger_v1.yaml")      # Modell (eval), Vorverarbeitung, Klassen
zeilen = [z.split(",") for z in MANIFEST.read_text().splitlines()[1:]]
val = [(pfad, int(label)) for s, pfad, label, md5 in zeilen if s == "val"]

pred = []
with torch.no_grad():
    for i in range(0, len(val), 64):                             # in Paketen zu 64 Bildern
        x = torch.stack([tf(Image.open(ROOT / p).convert("RGB")) for p, _ in val[i:i + 64]])
        pred += model(x).argmax(1).tolist()
y = np.array([l for _, l in val]); pred = np.array(pred)

recall = [float((pred[y == c] == c).mean()) for c in range(len(classes))]     # Trefferquote je wahrer Klasse
metriken = {"accuracy": round(float((pred == y).mean()), 4),
            "balanced_accuracy": round(float(np.mean(recall)), 4),
            "n_val": int(len(y)),
            "fehler": int((pred != y).sum()),
            "recall": {c: round(r, 4) for c, r in zip(classes, recall)}}
# Gegenprobe: dieselbe Genauigkeit wie beim Training (steht in der YAML)?
cfg = yaml.safe_load(Path("models/traeger_v1.yaml").read_text())
metriken["gleich_wie_train"] = bool(abs(cfg["metrics_val"]["accuracy"] - metriken["accuracy"]) < 1e-4)

Path("lab06/reports").mkdir(parents=True, exist_ok=True)
Path("lab06/reports/metrics.json").write_text(json.dumps(metriken, indent=1, ensure_ascii=False) + "\n", newline="\n")
print(f"evaluate: Genauigkeit {metriken['accuracy']:.2%} auf {len(y)} Bildern, {metriken['fehler']} Fehler | {time.perf_counter() - t0:.1f} s")
