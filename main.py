from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Data Monitoring Tool Running"}