import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import torch.nn.functional as F

# ---------------------------
# Configuration
# ---------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_NAME = "gpt2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROMPTS_PATH = Path("../data/prompts.json")
DOCS_DIR = Path("../data/docs/")

TOP_K = 1  # number of docs to retrieve per prompt

# ---------------------------
# Load prompts
# ---------------------------

with open(PROMPTS_PATH, "r") as f:
    prompts = json.load(f)

print('prompts loaded')