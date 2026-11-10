"""Stufe 2 von 4 der Traegermodell-Pipeline (AKE Lab 6): featurize.

Schickt jedes Bild einmal durch den eingefrorenen ResNet-18-Backbone (ohne Kopf) und speichert die
512-dimensionalen Merkmale - genau wie cached_features() in Lab 2, Aufgabe 4.2, gleiche Dateinamen:
  - lab06/cache/features_resnet18_160_train.pt
  - lab06/cache/features_resnet18_160_val.pt
Welche Bilder dazugehoeren, steht im Manifest (Ausgabe von prepare) - nicht in train.txt/val.txt.

Aufruf (aus dem Hauptordner des Uebungs-Repos):  python lab06/featurize.py
"""
import hashlib, os, sys, time
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "lab02")                          # damit "from src.traeger import ..." wie in Lab 2 funktioniert
from src.traeger import Imagenette5, make_transform, build_model

torch.set_num_threads(min(4, os.cpu_count()))       # Referenz: 4 Kerne -> vergleichbare Laufzeiten
torch.manual_seed(0)                                 # fester Zufall (hier eigentlich keiner noetig - sicher ist sicher)

ROOT = Path("data/imagenette5")
WEIGHTS = Path("data/models/resnet18-f37072fd.pth")
MANIFEST = Path("lab06/data/manifest.csv")
OUT = Path("lab06/cache"); OUT.mkdir(parents=True, exist_ok=True)


def lies_manifest(split):
    """Liste von (pfad, label) fuer einen Split, in der Reihenfolge des Manifests."""
    zeilen = [z.split(",") for z in MANIFEST.read_text().splitlines()[1:]]   # [1:] = Kopfzeile ueberspringen
    return [(pfad, int(label)) for s, pfad, label, md5 in zeilen if s == split]


weights_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()[:12]   # wie WEIGHTS_SHA in Lab 2
model = build_model(n_classes=5, backbone_weights=WEIGHTS)
backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten()).eval()   # alles ausser dem Kopf fc

for split in ("train", "val"):
    t0 = time.perf_counter()
    ds = Imagenette5(ROOT, split, make_transform(train=False))   # Vorverarbeitung ohne Augmentierung
    ds.items = lies_manifest(split)                              # Dateiliste aus dem Manifest (deklarierte Eingabe)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=64, num_workers=0):
            feats.append(backbone(x)); labels.append(y)
    d = {"x": torch.cat(feats), "y": torch.cat(labels), "weights_sha": weights_sha, "transform": repr(ds.transform)}
    # Keine Laufzeit und kein Datum in die Datei schreiben: Gleiche Eingaben sollen bitgleiche Ausgaben ergeben.
    torch.save(d, OUT / f"features_resnet18_160_{split}.pt")
    print(f"featurize {split}: {tuple(d['x'].shape)} | {time.perf_counter() - t0:.1f} s")
