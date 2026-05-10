import io
import base64
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, render_template, request

from TrashDetect.config import Config
from dataset import TrashDataset
from model import build_model

# ── Config ──────────────────────────────────────────────
MODEL_PATH  = "trained_models/best_model.pth"
DATA_PATH   = Config.DATA_TACO_PATH
IMAGE_SIZE  = 640
CONF_THRESH = 0.5

PALETTE = [
    (255,107,107),(255,217,61),(107,203,119),(77,150,255),(199,125,255),
    (244,162,97),(46,196,182),(231,111,81),(168,218,220),(247,37,133),
]

app = Flask(__name__)

# ── Load model ───────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ds         = TrashDataset(root=DATA_PATH, split="valid")
LABEL_MAP   = _ds.get_label_map()   # {1: "name", ...}
NUM_CLASSES = _ds.get_num_classes()

model = build_model(num_classes=NUM_CLASSES).to(device)
ckpt  = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"✅ Model loaded | device={device}")


# ── Helpers ──────────────────────────────────────────────
def predict(pil_img: Image.Image):
    img_r  = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = F.to_tensor(img_r).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)[0]

    boxes  = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()

    ow, oh = pil_img.size
    boxes[:, [0,2]] *= ow / IMAGE_SIZE
    boxes[:, [1,3]] *= oh / IMAGE_SIZE

    mask = scores >= CONF_THRESH
    return boxes[mask], labels[mask], scores[mask]


def draw_results(img, boxes, labels, scores):
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arialbd.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        r, g, b = PALETTE[int(label) % len(PALETTE)]
        draw.rectangle([x1,y1,x2,y2], outline=(r,g,b,255), width=3)
        draw.rectangle([x1,y1,x2,y2], fill=(r,g,b,40))

        text = f"{LABEL_MAP.get(int(label), label)} {score:.0%}"
        bb   = draw.textbbox((0,0), text, font=font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        p  = 4
        lx = max(x1, 0); ly = max(y1-th-p*2, 0)
        draw.rectangle([lx, ly, lx+tw+p*2, ly+th+p*2], fill=(r,g,b,220))
        draw.text((lx+p, ly+p), text, fill="white", font=font)
    return img


# ── Routes ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    pil_img = Image.open(request.files["image"].stream).convert("RGB")
    boxes, labels, scores = predict(pil_img)
    result   = draw_results(pil_img.copy(), boxes, labels, scores)

    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    detections = sorted(
        [{"class": LABEL_MAP.get(int(l), f"cls_{l}"), "score": float(s)}
         for l, s in zip(labels, scores)],
        key=lambda x: x["score"], reverse=True
    )

    return jsonify({"image": img_b64, "detections": detections})


if __name__ == "__main__":
    app.run(debug=True, port=5000)