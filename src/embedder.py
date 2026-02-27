from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    """
    A wrapper for sentence transformers to embed documents or prompts
    """
    def __int__(self, model_name: str):
        """
        Initialize the embedder with model name
        """
        self.model = SentenceTransformer(model_name)
