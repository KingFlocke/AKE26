"""Stufe 3 von 4 der Traegermodell-Pipeline (AKE Lab 6): train.

Zwei Schritte wie in Lab 2, Aufgaben 4.3-4.5 - alle Hyperparameter kommen aus params.yaml (Block "train"):
  1. Linearer Kopf 512 -> 5 auf den gecachten Merkmalen (Feature-Extraktion).
  2. Kopf + layer4 feinjustieren, auf den Bildern mit Augmentierung, BatchNorm eingefroren.
Ausgaben (Schema wie Lab 2, Aufgabe 4.5 und 5.2):
  - models/traeger_v1.pt    (State Dict)
  - models/traeger_v1.yaml  (Modellbeschreibung inkl. Validierungsmetriken)

Aufruf (aus dem Hauptordner des Uebungs-Repos):  python lab06/train.py
"""
import copy, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "lab02")
from src.traeger import (Imagenette5, make_transform, build_model, set_trainable, train_mode,
                         IMAGENET_MEAN, IMAGENET_STD, SIZE)

torch.set_num_threads(min(4, os.cpu_count()))
P = yaml.safe_load(Path("params.yaml").read_text())["train"]     # nur der Block "train" (so steht es in dvc.yaml)

ROOT = Path("data/imagenette5")
WEIGHTS = Path("data/models/resnet18-f37072fd.pth")
MANIFEST = Path("lab06/data/manifest.csv")
CACHE = Path("lab06/cache")
MODELS = Path("models")
lossf = nn.CrossEntropyLoss()


def lies_manifest(split):
    zeilen = [z.split(",") for z in MANIFEST.read_text().splitlines()[1:]]
    return [(pfad, int(label)) for s, pfad, label, md5 in zeilen if s == split]


def epoche(model, loader, opt=None):
    """Eine Epoche (opt=None: nur auswerten). Rueckgabe: (mittlerer Verlust, Genauigkeit)."""
    model.train(opt is not None)
    summe, richtig, n = 0.0, 0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            logits = model(x)
            loss = lossf(logits, y)
            if opt is not None:
                opt.zero_grad(); loss.backward(); opt.step()
            summe += loss.item() * len(y); richtig += (logits.argmax(1) == y).sum().item(); n += len(y)
    return summe / n, richtig / n


def trainiere(model, opt, loaders, epochs, name):
    """Trainiert und behaelt den Stand mit dem kleinsten Val-Verlust (wie fit(..., patience=epochs) in Lab 2)."""
    bester, stand, beste_ep = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = epoche(model, loaders["train"], opt)
        va = epoche(model, loaders["val"])
        print(f"  {name} Epoche {ep}: train acc {tr[1]:.2%} | val loss {va[0]:.4f} acc {va[1]:.2%} | {time.perf_counter() - t0:.1f} s")
        if va[0] < bester:
            bester, stand, beste_ep = va[0], copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(stand)
    return beste_ep


class BNFrozen(nn.Module):
    """.train() schaltet in den Trainingsmodus, laesst BatchNorm aber im eval()-Modus (wie in Lab 2)."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return self.m(x)
    def train(self, mode=True):
        super().train(mode)
        if mode:
            train_mode(self.m)
        return self


t_start = time.perf_counter()
# ---- Schritt 1: linearer Kopf auf den gecachten Merkmalen (Lab 2, Aufgabe 4.3) ----------------------------
feat = {s: torch.load(CACHE / f"features_resnet18_160_{s}.pt") for s in ("train", "val")}
dl_head = {"train": DataLoader(TensorDataset(feat["train"]["x"], feat["train"]["y"]), batch_size=P["head"]["batch"], shuffle=True),
           "val":   DataLoader(TensorDataset(feat["val"]["x"], feat["val"]["y"]), batch_size=256)}
torch.manual_seed(P["seed"])
head = nn.Linear(512, 5)
trainiere(head, torch.optim.Adam(head.parameters(), lr=P["head"]["lr"]), dl_head, P["head"]["epochs"], "Kopf")

# ---- Schritt 2: Kopf + layer4 feinjustieren (Lab 2, Aufgabe 4.4) ----------------------------------------
torch.manual_seed(P["seed"])
model = build_model(n_classes=5, backbone_weights=WEIGHTS)
model.fc.load_state_dict(head.state_dict())                     # trainierten Kopf uebernehmen
set_trainable(model, stages=("fc", "layer4"))
F = P["finetune"]
opt = torch.optim.Adam([{"params": model.fc.parameters(), "lr": F["lr_head"]},
                        {"params": model.layer4.parameters(), "lr": F["lr_backbone"]}])
ds_train = Imagenette5(ROOT, "train", make_transform(train=True)); ds_train.items = lies_manifest("train")
ds_val = Imagenette5(ROOT, "val", make_transform(train=False)); ds_val.items = lies_manifest("val")
dl_img = {"train": DataLoader(ds_train, batch_size=F["batch"], shuffle=True, num_workers=0),
          "val":   DataLoader(ds_val, batch_size=64, num_workers=0)}
torch.manual_seed(P["seed"])
beste_ep = trainiere(BNFrozen(model), opt, dl_img, F["epochs"], "Fine-Tuning")

# ---- Validierungsmetriken (wie Lab 2, Aufgabe 5.1/5.2) -----------------------------------------------------
model.eval()
with torch.no_grad():
    pred = torch.cat([model(x).argmax(1) for x, _ in dl_img["val"]]).numpy()
y_val = np.array([y for _, y in ds_val.items])
M = np.zeros((5, 5), dtype=int)
for t, p in zip(y_val, pred):
    M[t, p] += 1
prec = np.diag(M) / np.maximum(M.sum(axis=0), 1)
rec = np.diag(M) / np.maximum(M.sum(axis=1), 1)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)

# ---- Ablegen: State Dict + Modellbeschreibung (Schema aus models/traeger_v1.yaml.template) ------------------
MODELS.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODELS / "traeger_v1.pt")
cfg = yaml.safe_load((MODELS / "traeger_v1.yaml.template").read_text())
cfg["architecture"]["params_total"] = sum(p.numel() for p in model.parameters())
cfg["architecture"]["backbone_weights_sha256_12"] = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()[:12]
cfg["classes"] = [{"index": c["index"], "name": c["name"], "wnid": c["wnid"]}
                  for c in json.loads((ROOT / "classes.json").read_text())]
cfg["preprocessing"].update(resize_short_side=SIZE, center_crop=SIZE, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
cfg["data"]["n_train"], cfg["data"]["n_val"] = len(ds_train), len(ds_val)
cfg["training"]["seed"] = P["seed"]
cfg["training"]["torch"] = str(torch.__version__)
cfg["training"]["stage1_feature_extraction"].update(lr=P["head"]["lr"], epochs=P["head"]["epochs"], batch=P["head"]["batch"])
cfg["training"]["stage2_finetune"].update(lr_head=F["lr_head"], lr_backbone=F["lr_backbone"], epochs=F["epochs"],
                                          batch=F["batch"], best_epoch=beste_ep)
cfg["metrics_val"].update(accuracy=round(float(np.trace(M) / M.sum()), 4), balanced_accuracy=round(float(rec.mean()), 4),
                          macro_f1=round(float(f1.mean()), 4), precision=[round(float(v), 4) for v in prec],
                          recall=[round(float(v), 4) for v in rec], confusion_matrix=M.tolist())
# Bewusst KEIN Tagesdatum: sonst haette jede Ausgabe einen anderen Hash, obwohl sich nichts geaendert hat.
cfg["created"] = "siehe git log (Commit mit dvc.lock)"
(MODELS / "traeger_v1.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), newline="\n")
print(f"train: Val-Genauigkeit {cfg['metrics_val']['accuracy']:.2%} (beste Epoche {beste_ep}) | {time.perf_counter() - t_start:.1f} s")
