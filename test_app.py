# pip install fastapi uvicorn
# uvicorn main:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="FastAPI Test Server")


@app.get("/")
def root():
    return {
        "status": "success",
        "message": "FastAPI is running successfully!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/hello/{name}")
def hello(name: str):
    return {
        "message": f"Hello {name}!"
    }


class User(BaseModel):
    name: str
    age: int


@app.post("/user")
def create_user(user: User):
    return {
        "message": "User received successfully",
        "data": user
    }
