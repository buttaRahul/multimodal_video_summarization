from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from helper import generate_response
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)



class URLRequest(BaseModel):
    url: str

@app.post("/submit-url/")
async def submit_url(request: URLRequest):
    print("URL received")
    response = generate_response(request.url)
    return {
        "message": "URL received successfully",
        "url": request.url,
        "summary_response": response["summary"],
        "frames": response["frames"]
    }