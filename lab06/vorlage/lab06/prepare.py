"""Stufe 1 von 4 der Traegermodell-Pipeline (AKE Lab 6): prepare.

Liest die Dateilisten train.txt und val.txt, prueft jede Bilddatei und schreibt
  - lab06/data/manifest.csv      (eine Zeile je Bild: split, pfad, label, md5)  -> Eingabe fuer featurize, train, evaluate
  - lab06/reports/datencheck.json (Pruefbericht: Anzahl, defekte Dateien, Duplikate, Bildgroessen, Klassen je Split)

Entspricht Aufgabe 0 aus Lab 2, nur als eigenes Skript mit festen Ein- und Ausgaben.
Aufruf (aus dem Hauptordner des Uebungs-Repos):  python lab06/prepare.py
"""
import collections, hashlib, json, time
from pathlib import Path
from PIL import Image

t0 = time.perf_counter()
ROOT = Path("data/imagenette5")                     # Eingabe: der Datensatz (von DVC verwaltet)
OUT = Path("lab06")                                 # Ausgaben landen unter lab06/data und lab06/reports

zeilen = []                                         # Zeilen des Manifests
defekt = 0                                          # Zaehler fuer unlesbare Dateien
groessen = collections.Counter()                    # z. B. {"160x160": 3000}
klassen = collections.Counter()                     # z. B. {"train/0": 500, ...}

for split in ("train", "val"):
    for line in (ROOT / f"{split}.txt").read_text().splitlines():
        rel, y = line.split("\t")                   # "val/n03028079/ILSVRC2012_val_00008116.JPEG", "1"
        klassen[f"{split}/{y}"] += 1
        try:
            with Image.open(ROOT / rel) as im:
                groessen[f"{im.width}x{im.height}"] += 1
                im.verify()                         # Datei lesbar und vollstaendig?
        except Exception:
            defekt += 1
        md5 = hashlib.md5((ROOT / rel).read_bytes()).hexdigest()   # Pruefsumme des Inhalts (wie DVC)
        zeilen.append(f"{split},{rel},{y},{md5}")

# Manifest schreiben. newline="\n": gleiche Bytes unter Windows, macOS und Linux -> gleicher Hash fuer DVC
(OUT / "data").mkdir(parents=True, exist_ok=True)
(OUT / "data/manifest.csv").write_text("split,pfad,label,md5\n" + "\n".join(zeilen) + "\n", newline="\n")

# Duplikate: gleiche Pruefsumme = gleicher Inhalt, egal wie die Datei heisst
md5s = [z.rsplit(",", 1)[1] for z in zeilen]
duplikate = len(md5s) - len(set(md5s))

bericht = {"n": len(zeilen), "defekt": defekt, "duplikate": duplikate,
           "groessen": dict(groessen), "klassen": dict(sorted(klassen.items()))}
(OUT / "reports").mkdir(exist_ok=True)
(OUT / "reports/datencheck.json").write_text(json.dumps(bericht, indent=1) + "\n", newline="\n")
print(f"prepare: {len(zeilen)} Bilder, {defekt} defekt, {duplikate} Duplikate | {time.perf_counter() - t0:.1f} s")
