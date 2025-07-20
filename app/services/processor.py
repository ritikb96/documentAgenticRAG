from typing import Dict, Any, List
from app.services.title_summary import ChunkSummarizer
from app.services.embedder import Embedder
from dataclasses import dataclass
import asyncio
import uuid

embedder = Embedder()
chunker = ChunkSummarizer()

@dataclass
class ProcessedChunk:
    id:int
    chunk_number:int
    title:str
    summary:str
    content:str
    filename:str
    chunking_method:str
    metadata : Dict[str, Any]
    embedding : List[float]
    embedding_model:str


async def process_single_chunk(
    chunk: str,
    chunk_number: int,
    filename:str,
    chunking_method:str,
) -> ProcessedChunk:
    try:
       extracted = await chunker.get_title_and_summary(chunk)
    except Exception as e:
        print(f"Failed to summarize chunk {chunk_number}: {e}")
        extracted = {"title": "Error", "summary": "Could not generate summary"}

    try:
        embedding_dict = await embedder.get_embedding(chunk)
        embedding = embedding_dict['embedding']
        embedding_model = embedding_dict['embedding_model']
    except Exception:
        embedding = [0.0] * 1536  
        embedding_model = "none"
        


    return ProcessedChunk(
            id = str(uuid.uuid4()),
            chunk_number = chunk_number,
            title = extracted['title'],
            summary=  extracted['summary'],
            content=  chunk,
            filename=  filename,
            chunking_method = chunking_method,
            metadata =  {},
            embedding =  embedding,
            embedding_model= embedding_model
    )
 