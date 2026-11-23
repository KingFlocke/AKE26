# Vorberechnete Logs (AKE Lab 8)

`log_referenz.npz` und `log_drift.npz` enthalten die beiden Fenster des Referenzlaufs: `p` (501 × 5 Softmax-Wahrscheinlichkeiten), `emb` (501 × 512 Embedding-Werte, Schicht `avgpool` von ResNet-18) und `labels` (richtige Klasse je Bild, Reihenfolge wie `data/imagenette5/val.txt`).

Das Notebook lädt sie in Aufgabe 2.4 automatisch, wenn der Dienst auf Ihrem Rechner nicht läuft. Driftstrom: Helligkeit × 1,6 · Gaussian Blur (5 Pixel, σ = 1,5) · JPEG-Qualität 20.
