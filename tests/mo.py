from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
print("Model loaded successfully!")
