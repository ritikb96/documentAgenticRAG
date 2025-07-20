from app.services.processor import ProcessedChunk
from app.models.booking import booking_data
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from datetime import datetime,timezone
from datetime import date,time,datetime
from pydantic import BaseModel,EmailStr



load_dotenv()

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

async def insert_chunk(chunk: ProcessedChunk):
    """Insert chunk metadata into the Supabase database."""
    try:
        data = {
            "chunk_id": chunk.id,  # UUID generated earlier
            "chunk_number": chunk.chunk_number,
            "filename": chunk.filename,
            "chunking_method": chunk.chunking_method,
            "embedding_model": chunk.embedding_model,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        result = supabase.table("document_chunks").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for file '{chunk.filename}'")
        return result
    except Exception as e:
        print(f" Error inserting chunk: {e}")
        return None
    
def insert_bookingData(booking_Data: booking_data):
    """Insert chunk metadata into the Supabase database."""
    try:
        data = {
                "id": str(booking_Data.id),
                "full_name": booking_Data.full_name,
                "email": booking_Data.email  ,
                "interview_date": booking_Data.interview_date.isoformat()  ,
                "interview_time":booking_Data.time.isoformat(),
                'created_at': booking_Data.created_at.isoformat(),


        }

        result = supabase.table("interview_bookings").insert(data).execute()
        print(f" stored information for interviewee: {booking_Data.full_name}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

