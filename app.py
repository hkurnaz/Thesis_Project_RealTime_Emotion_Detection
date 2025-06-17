from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from facenet_pytorch import MTCNN
from PIL import Image
import io, torch, timm
from torchvision import transforms
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1) Sınıf etiketleri
classes = ['anger','disgust','fear','happiness','neutrality','sadness','surprise']

# 2) Modeli yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load("best.pt", map_location=device))
model.to(device).eval()

# 3) Yüz detektörü (birden fazla yüz için)
mtcnn = MTCNN(keep_all=True, device=device)

# 4) Görüntü ön‐işlem (test kodundakiyle tamamen aynı)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(img)

    # yüzleri tespit et
    boxes, _ = mtcnn.detect(rgb)
    preds = []
    if boxes is not None:
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            face = rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue

            x = preprocess(face).unsqueeze(0).to(device)
            with torch.no_grad():
                label_idx = model(x).argmax(1).item()
            preds.append({
                "id": idx + 1,
                "box": [x1, y1, x2, y2],
                "label": classes[label_idx]
            })

    return {"predictions": preds}

# frontend klasörünü ana rota olarak yayınla
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

#uvicorn app:app --reload