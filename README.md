# production-ready rag pipeline with llm fine-tuning for domain specific question-answering

this project implements a retrieval-augmented generation (rag) pipeline combined with lightweight llm fine-tuning for domain-specific question answering.

it integrates document embeddings, vector similarity search, and a fine-tuned language model to generate context-aware responses.

the system demonstrates practical genai architecture used in modern enterprise ai systems.

## architecture

- sentence-transformers for document embeddings
- faiss for vector similarity search
- huggingface transformers for llm fine-tuning
- fastapi for serving inference api
- modular training and inference scripts

## features

- document embedding and indexing
- vector similarity retrieval
- lightweight llm fine-tuning
- domain-specific qa generation
- api-based inference deployment
- modular, production-style structure

## project structure

rag-llm-pipeline/
│
├── data/
│   └── domain_qa.json
│
├── models/
│   └── (fine-tuned model checkpoints)
│
├── train.py
├── embed.py
├── app.py
├── requirements.txt
└── README.md

## how to run locally

### 1. clone the repository

git clone <your-repo-url>
cd rag-llm-pipeline

### 2. install dependencies

pip install -r requirements.txt

### 3. prepare dataset

place your domain-specific qa dataset inside the data/ directory.
expected format example:

[
  {"text": "document content here", "label": "target output"},
  {"text": "another document", "label": "target output"}
]

### 4. train / fine-tune the model

python train.py

this saves fine-tuned model checkpoints inside the models/ directory.

### 5. build vector index

python embed.py

this creates embeddings and initializes the faiss index for retrieval.

### 6. run inference api

uvicorn app:app --reload

api runs at:
http://127.0.0.1:8000

example request:

GET /generate?prompt=your_question_here

## retrieval flow

1. embed incoming query
2. retrieve top-k relevant documents from faiss
3. augment prompt with retrieved context
4. generate final response using fine-tuned llm

## use cases

- enterprise internal knowledge assistants
- legal or financial domain qa
- research summarization
- regulatory documentation search
- customer support automation

## future improvements

- quantized model deployment
- hybrid bm25 + vector search
- streaming responses
- evaluation pipeline (bleu / rouge / exact match)
- dockerized multi-service deployment

---

this project demonstrates practical understanding of rag architecture, vector databases, llm fine-tuning workflows, and production-style ai system design.
