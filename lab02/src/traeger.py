"""Traegermodell (AKE): Daten, Vorverarbeitung, Modell und Laden aus models/traeger_v1.{pt,yaml}.

Dieses Modul ist die Schnittstelle, die in den Skripten Daten-Engineering fuer ML, MLOps: Vom Notebook zum
Produktivsystem, Monitoring & Drift und Erklaerbare KI & Compliance weiterverwendet wird. Nichts hier
haengt vom Notebook ab; Klassenreihenfolge und Vorverarbeitung kommen aus der YAML-Datei, nie aus os.listdir.

Aufruf aus der Kommandozeile (Aufgabe 5, frische Python-Sitzung):
    python -m src.traeger models/traeger_v1.yaml <bild.jpg>
"""
import json, sys
from pathlib import Path

import torch, torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIZE = 160


# --------------------------------------------------------------------------------------- Daten
class Imagenette5(Dataset):
    """Kuratierte Imagenette-Teilmenge; Split = Dateiliste <root>/<split>.txt (Pfad<TAB>Klassenindex)."""

    def __init__(self, root, split, transform=None):
        self.root, self.transform = Path(root), transform
        lines = (self.root / f"{split}.txt").read_text().strip().splitlines()
        self.items = [(l.split("\t")[0], int(l.split("\t")[1])) for l in lines]
        self.classes = [c["name"] for c in json.loads((self.root / "classes.json").read_text())]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y = self.items[i]
        img = Image.open(self.root / rel).convert("RGB")
        return (self.transform(img) if self.transform else img), y


def make_transform(train=False, normalize=True, size=SIZE):
    """Eval: kuerzere Seite -> size, Center-Crop, ImageNet-Normalisierung (Skript Abschn. 5.2, Detail 1).
    Train: RandomResizedCrop, HorizontalFlip, ColorJitter (Skript Abschn. 3.4). normalize=False nur fuer
    das Zusatzexperiment in Aufgabe 3."""
    aug = ([transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2)]
           if train else [transforms.Resize(size), transforms.CenterCrop(size)])
    norm = [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)] if normalize else []
    return transforms.Compose(aug + [transforms.ToTensor()] + norm)


# --------------------------------------------------------------------------------------- Modell
def find_backbone_weights(models_dir):
    """ResNet-18-ImageNet-Gewichte vom Hochschul-Share (resnet18-f37072fd.pth = torchvision IMAGENET1K_V1).
    Liegt eine andere resnet18*.pth-Datei dort, wird sie genommen (und ihr Name im Log vermerkt)."""
    models_dir = Path(models_dir)
    ref = models_dir / "resnet18-f37072fd.pth"
    if ref.exists():
        return ref
    cands = sorted(models_dir.glob("resnet18*.pth"))
    return cands[0] if cands else None


def build_model(n_classes=5, backbone_weights=None):
    """ResNet-18, Kopf 512 -> n_classes (Skript Listing 3). backbone_weights: Pfad zum State Dict (1000 Klassen)
    oder None -> torchvision laedt IMAGENET1K_V1 herunter (nur Fallback, ~45 MB)."""
    if backbone_weights is not None:
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(backbone_weights, map_location="cpu"))
    else:
        from torchvision.models import ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def set_trainable(model, stages=("fc",)):
    """requires_grad nur fuer Parameter, deren Name mit einem der Praefixe beginnt (z. B. ("fc", "layer4"))."""
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(s) for s in stages)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_mode(model):
    """train() fuer Dropout/Grad, aber alle BatchNorm-Schichten im eval()-Modus: laufende Statistiken
    bleiben die aus ImageNet (Skript Abschn. 5.2, Detail 2; Listing 3)."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return model


# --------------------------------------------------------------------------------------- Laden / Inferenz
def load_traeger(yaml_path):
    """Modell + Vorverarbeitung + Klassenliste aus traeger_v1.yaml und dem daneben liegenden State Dict."""
    yaml_path = Path(yaml_path)
    cfg = yaml.safe_load(yaml_path.read_text())
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(cfg["classes"]))
    model.load_state_dict(torch.load(yaml_path.parent / cfg["state_dict"], map_location="cpu"))
    model.eval()
    pre = cfg["preprocessing"]
    tf = transforms.Compose([transforms.Resize(pre["resize_short_side"]),
                             transforms.CenterCrop(pre["center_crop"]),
                             transforms.ToTensor(),
                             transforms.Normalize(pre["mean"], pre["std"])])
    classes = [c["name"] if isinstance(c, dict) else c for c in cfg["classes"]]   # Liste von Dicts (index, name, wnid)
    assert [c.get("index", i) if isinstance(c, dict) else i for i, c in enumerate(cfg["classes"])] == list(range(len(classes)))
    return model, tf, classes


@torch.no_grad()
def predict(model, tf, classes, image):
    """image: Pfad oder PIL-Bild. Rueckgabe: (Klassenname, Softmax-Wahrscheinlichkeiten als Liste).
    Das Modell gibt Logits aus; Softmax erst hier, d. h. im 'Service' (Skript Abschn. 5.4)."""
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    logits = model(tf(image).unsqueeze(0))
    p = logits.softmax(1)[0]
    return classes[int(p.argmax())], p.tolist()


if __name__ == "__main__":
    model, tf, classes = load_traeger(sys.argv[1])
    for img in sys.argv[2:]:
        name, p = predict(model, tf, classes, img)
        print(f"{img}: {name}  " + " ".join(f"{c}={v:.3f}" for c, v in zip(classes, p)))
