# app/services/embedder.py

import os
from typing import List
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Embedder:
    def __init__(self, model: str = "text-embedding-3-small", fallback_dim: int = 1536):
        self.model = model
        self.fallback_dim = fallback_dim
        self.embedding_model = str(model)

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from OpenAI for a given text."""
        try:
            response = await openai_client.embeddings.create(
                model=self.model,
                input=text
            )
            return {
                "embedding":response.data[0].embedding,
                "embedding_model":self.embedding_model
            }
        except Exception as e:
            print(f"Error embedding the text: {e}")
            return [0.0] * self.fallback_dim
