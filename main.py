from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io

# 1. Initialize FastAPI app
app = FastAPI(title="Luminary Identity Verification API")
device = torch.device("cpu")

# ==========================================
# 2. YOUR EXACT CLASS FROM THE NOTEBOOK
# ==========================================
class FaceEmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128, use_pretrained=False):
        super().__init__()

        if use_pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_size)

    def forward(self, x):
        embeddings = self.backbone(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

# ==========================================
# 3. LOAD THE MODEL WEIGHTS
# ==========================================
# Initialize the class with your parameters
model = FaceEmbeddingNet(embedding_size=128, use_pretrained=False)

# Load the .pth file (the 'suitcase')
MODEL_PATH = "luminary_identity_arcface_final.pth.zip"

# Note: weights_only=False is required here because your checkpoint contains strings and dictionaries (like identity_names)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Extract JUST the model weights from the bundle and load them
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ==========================================
# 4. YOUR EXACT TRANSFORM FROM THE NOTEBOOK
# ==========================================

verification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ==========================================
# 5. THE API ENDPOINT
# ==========================================
@app.post("/verify")
async def verify_identity(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Validate file types
    if not file1.content_type.startswith("image/") or not file2.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Both files must be images.")
        
    try:
        # Read and open the uploaded images
        img1 = Image.open(io.BytesIO(await file1.read())).convert('RGB')
        img2 = Image.open(io.BytesIO(await file2.read())).convert('RGB')
        
        # Apply your transformations and add the batch dimension
        img1_tensor = verification_transform(img1).unsqueeze(0).to(device)
        img2_tensor = verification_transform(img2).unsqueeze(0).to(device)
            
        # Extract features (embeddings)
        with torch.no_grad():
            emb1 = model(img1_tensor)
            emb2 = model(img2_tensor)
            
            # Calculate Cosine Similarity
            similarity = F.cosine_similarity(emb1, emb2).item()
            
            # Your specific threshold from the notebook
            threshold = 0.4
            same_person = bool(similarity > threshold)
            
        return {
            "file1_name": file1.filename,
            "file2_name": file2.filename,
            "similarity_score": round(similarity, 4),
            "prediction": "SAME PERSON" if same_person else "DIFFERENT PERSON",
            "is_match": same_person,
            "threshold_used": threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Luminary Identity API is running!"}