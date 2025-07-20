# email.py

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from app.models.booking import booking_data

load_dotenv()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")  
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECEIVER = os.getenv("RECEIVER_EMAIL")

def notify_admin_booking(data:booking_data):
    subject = "New Interview Booking Received"

    body = f"""Dear DocuRAG Admin,

    A new interview has been booked.

    👤 Name: {data.full_name}  
    📧 Email: {data.email}  
    📅 Date: {data.interview_date}  
    ⏰ Time: {data.time}

    Please check the database for full details.

    Regards,  
    DocuRAG Agent"""

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS,"byeolsujengrg@gmail.com", msg.as_string())
        print("Notification email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")
