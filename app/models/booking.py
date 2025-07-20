from pydantic import BaseModel, EmailStr
from datetime import date, time, datetime

class booking_data(BaseModel):
    id: str
    full_name: str
    email: EmailStr
    interview_date: date
    time: time
    created_at: datetime
