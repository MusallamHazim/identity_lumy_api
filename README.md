# Luminary Identity Verification API

ResNet18 face-embedding model (ArcFace training) that verifies whether two face photos belong to the same person. Part of the Luminary (Lumy) graduation project.

## Files
- `main.py` — FastAPI app
- `luminary_identity_arcface_final.pth.zip` — trained model checkpoint

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Then open `http://127.0.0.1:8000/docs` to test.

## Endpoint
- `POST /verify` — upload two face images, returns similarity score + match decision.
