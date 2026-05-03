# Custom Q&A Bot (RAG-based)

This project demonstrates how a Retrieval-Augmented Generation (RAG) system works to generate accurate, context-aware responses. It integrates document retrieval with a Large Language Model (LLM) to answer user queries effectively.

The project uses the **Claude Sonnet 4.5** model for generating responses.

---

## Features

* End-to-end RAG pipeline implementation
* Document ingestion and embedding using vector database
* FastAPI-based backend
* Interactive UI for asking questions
* PDF output generation

---

## Tech Stack

* Python
* FastAPI
* ChromaDB (Vector Database)
* Sentence Transformers (Embeddings)
* Anthropic Claude API
* Pandas

---

## ⚙️ Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Environment

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## API Key Setup

Get your API key from Anthropic (or any supported LLM provider) and set it as an environment variable:

```bash
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

---

## Run the Application

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Open in Browser:

```
http://localhost:8000
```

---

## Troubleshooting

If you encounter package installation issues, try:

```bash
pip install --upgrade pip
pip install pandas --prefer-binary
pip install fastapi uvicorn python-multipart chromadb sentence-transformers openpyxl anthropic
```

---

##  UI Output

<img width="1904" height="956" alt="image" src="https://github.com/user-attachments/assets/d962b520-0867-42f3-963b-6c4ff8de485f" />


## PDF Output



<img width="1910" height="956" alt="image" src="https://github.com/user-attachments/assets/60825c63-a8bb-43f4-b30f-6a8eedbbe310" />

---

## Project Purpose

The main goal of this project is to:

* Understand how RAG pipelines work
* Combine retrieval with LLMs for better accuracy
* Build a real-world Q&A system

---

## Notes

* Make sure your API key is valid and active
* Ensure all dependencies are installed properly
* This project can be extended with other LLMs like OpenAI or local models

---

