import os
import numpy as np
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from flask import Flask, request, jsonify

# --- Configuration & Setup ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- NEW: Load the API Key from Environment Variables ---
# This is the secret key your server expects.
EXPECTED_API_KEY = os.environ.get("MY_API_KEY")
if not EXPECTED_API_KEY:
    # This will stop the server from starting if the key is not set.
    raise ValueError("FATAL ERROR: MY_API_KEY environment variable is not set.")

# --- Function Definitions (No Changes Here) ---
def load_pdf(path):
    print(f"Loading PDF from: {path}")
    if not os.path.exists(path):
        return None
    text = ""
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def get_embeddings(chunks, model):
    return np.array(model.encode(chunks, show_progress_bar=False)).astype("float32")

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_index(query, embed_model, index, chunks, k=3):
    query_embedding = embed_model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype("float32")
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def generate_answer(context, question, llm):
    prompt = f"""
Use the following context to answer the question at the end. If the answer is not contained
within the text below, say "I don't have enough information to answer this question."

### Context:
{context}

### Question:
{question}

### Answer:
"""
    output = llm(prompt, max_tokens=512, stop=["###", "Question:"])
    return output['choices'][0]['text'].strip()

# --- Global Variables & One-Time Startup ---
print("--- INITIALIZING APPLICATION ---")
PDF_PATH = "data/Arogya_Sanjeevani_Policy.pdf"
MODEL_PATH = "model/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

DOCUMENT_TEXT = load_pdf(PDF_PATH)
if DOCUMENT_TEXT is None:
    raise Exception("Failed to load PDF. Shutting down.")
CHUNKS = split_text(DOCUMENT_TEXT)

print("Loading embedding model...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDINGS = get_embeddings(CHUNKS, EMBED_MODEL)
INDEX = build_faiss_index(EMBEDDINGS)

print("Loading language model...")
LLM = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6, verbose=False)

print("\n✅ System ready! API is online and waiting for requests.")

# --- Flask Web Application ---
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    # --- NEW: API Key Verification ---
    # Get the API key from the request headers
    client_api_key = request.headers.get("X-API-Key")

    # Check if the key is missing or incorrect
    if client_api_key != EXPECTED_API_KEY:
        # If the key is wrong, reject the request
        return jsonify({"error": "Unauthorized"}), 401
    
    # --- If key is correct, proceed with the original logic ---
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    query = data['question']
    print(f"\nReceived authorized question: {query}")
    
    relevant_chunks = search_index(query, EMBED_MODEL, INDEX, CHUNKS)
    context = "\n---\n".join(relevant_chunks)
    answer = generate_answer(context, query, LLM)
    
    return jsonify({"answer": answer})

@app.route('/')
def health_check():
    return "RAG API is alive and running!"