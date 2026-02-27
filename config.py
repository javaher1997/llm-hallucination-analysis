from pathlib import Path
import torch

# ---------------------------
# Device
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------
# Model Names
# ---------------------------
MODEL_NAME = "gpt2"  # LLM model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # sentence embeddings

# ---------------------------
# Paths
# ---------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()  # current project root
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_PATH = DATA_DIR / "prompts.json"
DOCS_DIR = DATA_DIR / "docs"

# Optional output folder for embeddings, results, etc.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------
# Hyperparameters
# ---------------------------
TOP_K = 1               # number of documents to retrieve per prompt
MAX_NEW_TOKENS = 50     # max tokens to generate per prompt