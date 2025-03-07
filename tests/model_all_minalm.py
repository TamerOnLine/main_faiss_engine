import os
from sentence_transformers import SentenceTransformer

MODEL_PATH = "all-MiniLM-L6-v2"

try:
    model = SentenceTransformer(MODEL_PATH)
    print("✅ Model loaded successfully, either from local storage or the internet.")

    # Attempt to retrieve model directory
    model_dir = model._modules.get("0", None)
    model_dir_path = getattr(model_dir, "auto_model", None)
    if model_dir_path:
        print(f"🔍 Model loaded from path: {model_dir_path.config.name_or_path}")
    else:
        print("⚠️ Unable to determine model storage path.")

except Exception as e:
    print(f"❌ Error occurred while loading the model: {e}")

# Fallback: Print expected directory based on model name
print(f"🔍 Expected model directory: {os.path.abspath(MODEL_PATH)}")
