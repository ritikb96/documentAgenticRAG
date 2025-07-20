from fastapi import FastAPI,APIRouter
from app.api import document
from app.api import chat


app = FastAPI()

app.include_router(document.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"message":"you are are in the root "}