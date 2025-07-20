from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.upload import UploadService
from app.services.chunker import semantic_chunk_text
from app.services.processor import process_single_chunk, ProcessedChunk
from app.services.supabase_handler import insert_chunk
from app.services.qdrant_handler import store_embeddings

from qdrant_client.models import PointStruct
import asyncio
import uuid

router = APIRouter()
upload_service = UploadService()

# semaphore used here to limit concurrency
semaphore = asyncio.Semaphore(10) 

@router.post("/upload/")
async def upload_and_process(file: UploadFile = File(...)) -> dict:
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are allowed")

    try:
        # Step 1: Read + parse
        content = await file.read()
        parsed_content = await upload_service.parse_file(file.filename, content)

        # Step 2: Chunk the content
        chunk_result = semantic_chunk_text(parsed_content)
        chunk_texts = chunk_result["chunks"]
        chunk_method = chunk_result["chunking_method"]

        # Step 3: Limit concurrency using the semaphore
        async def limited_process(chunk, i):
            async with semaphore:
                return await process_single_chunk(
                    chunk=chunk,
                    chunk_number=i,
                    filename=file.filename,
                    chunking_method=chunk_method
                )

        tasks = [
            limited_process(chunk, i)
            for i, chunk in enumerate(chunk_texts)
        ]
        processed_chunks = await asyncio.gather(*tasks)


        #store embedding in qdrant
        #store  in supabase
        insert_chunks = [insert_chunk(chunk) for chunk in processed_chunks]
        store_chunks = [store_embeddings(chunk) for chunk in processed_chunks]
        await asyncio.gather(*insert_chunks,*store_chunks)


        return {
            "message":"added to all the databases successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
