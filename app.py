from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
generator = pipeline("text-generation", model="./models")

@app.get("/generate/")
def generate(prompt: str):
    output = generator(prompt, max_length=100)
    return {"response": output[0]["generated_text"]}
