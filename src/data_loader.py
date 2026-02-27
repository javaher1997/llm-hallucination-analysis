from config import PROMPTS_PATH, DOCS_DIR
from pathlib import Path
import json

def load_prompts(path=PROMPTS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_docs(docs_dir=DOCS_DIR):
    doc_paths = sorted(docs_dir.glob("*.txt"))
    docs = [p.read_text(encoding="utf-8") for p in doc_paths]
    return docs, doc_paths