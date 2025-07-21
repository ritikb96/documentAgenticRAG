from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.services.processor import ProcessedChunk
import asyncio
import os

client = AsyncQdrantClient(url="http://localhost:6333")


async def create_collection():
    if not await client.collection_exists(collection_name="docuRAG-embeddings"):
        await client.create_collection(
            collection_name="docuRAG-embeddings",
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE
            )
        )


async def store_embeddings(chunk: ProcessedChunk):
    try:
        await client.upsert(
            collection_name="docuRAG-embeddings",
            wait=True,
            points=[
                PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload={
                        "chunk_number": chunk.chunk_number,
                        "filename": chunk.filename,
                        "embedding_model": chunk.embedding_model,
                        "page_content": chunk.content
                    }
                )
            ]
        )
        print(f"Stored embedding for chunk: {chunk.chunk_number} for file '{chunk.filename}'")
    except Exception as e:
        print(f" Error storing embedding: {e}")
        return None

# asyncio.run(create_collection())
