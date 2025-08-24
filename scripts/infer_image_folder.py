import os
import torch
from PIL import Image
from torchvision import transforms
from open_clip import create_model_and_transforms, get_tokenizer

# --- Load model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai",
    device=device
)

tokenizer = get_tokenizer("ViT-B-32")

# --- Inference Labels ---
texts = tokenizer(["a baby", "a dog", "a toy", "a bottle", "a person"]).to(device)

# --- Image Folder Path ---
img_folder = "path/to/image/folder"

# --- Iterate over all images in folder ---
for fname in os.listdir(img_folder):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(img_folder, fname)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)
        logits = (100.0 * image_features @ text_features.T)
        probs = logits.softmax(dim=-1).cpu().numpy()

    print(f"Image: {fname}")
    for label, prob in zip(["baby", "dog", "toy", "bottle", "person"], probs[0]):
        print(f"  {label}: {prob:.4f}")