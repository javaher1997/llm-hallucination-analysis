# hallucination_pipeline.py
from config import DEVICE, PROMPTS_PATH, DOCS_DIR, EMBEDDING_MODEL_NAME
from src.data_loader import load_prompts, load_docs
from src.embedder import Embedder

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
    # Initialize the embedder and embed docs
    # ---------------------------
    print(f'Initializing embedder with model: {EMBEDDING_MODEL_NAME}')
    embedder = Embedder(EMBEDDING_MODEL_NAME)
    embedder.model.to(DEVICE)

    print('Embedding docs ...')
    doc_embeddings = embedder.model.encode(docs, convert_to_tensor=True, device=DEVICE)
    print(f'Document embedding shape: {doc_embeddings.shape}') # [num_docs, Embedding_dim]

    # ---------------------------
    # Placeholder for next steps
    # ---------------------------
    print("\nNext steps: embedding docs, retrieving relevant context, generating answers...")

if __name__ == "__main__":
    main()