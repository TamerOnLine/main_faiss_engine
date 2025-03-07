from transformers import AutoTokenizer

model_name = "all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")

text = "FAISS is a powerful library for similarity search."
tokens = tokenizer.tokenize(text)
print(tokens)
