# hallucination_pipeline.py
from config import DEVICE, PROMPTS_PATH, DOCS_DIR
from src.data_loader import load_prompts, load_docs

def main():
    print(f"Running pipeline on device: {DEVICE}")

    # ---------------------------
    # Load prompts
    # ---------------------------
    prompts = load_prompts(PROMPTS_PATH)
    print(f"Loaded {len(prompts)} prompts:")
    for p in prompts:
        print(f"  - {p['prompt']} (answer: {p['answer']})")

    # ---------------------------
    # Load docs
    # ---------------------------
    docs, doc_paths = load_docs(DOCS_DIR)
    print(f"Loaded {len(docs)} documents:")
    for dp in doc_paths:
        print(f"  - {dp.name}")

    # ---------------------------
    # Placeholder for next steps
    # ---------------------------
    print("\nNext steps: embedding docs, retrieving relevant context, generating answers...")

if __name__ == "__main__":
    main()