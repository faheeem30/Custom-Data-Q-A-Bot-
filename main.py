from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import anthropic
import uuid
import os
import shutil
import json

app = FastAPI(title="Custom Data Q&A Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(
    name="rag_data", embedding_function=ef, metadata={"hnsw:space": "cosine"}
)
anthropic_client = anthropic.Anthropic()

UPLOAD_DIR = "./uploads"
META_FILE = "./file_meta.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = ('.csv', '.xlsx', '.xls', '.pdf', '.txt', '.json', '.docx')


def load_meta() -> dict:
    if os.path.exists(META_FILE):
        with open(META_FILE) as f:
            return json.load(f)
    return {}


def save_meta(meta: dict):
    with open(META_FILE, "w") as f:
        json.dump(meta, f)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split long text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def parse_file(path: str, filename: str) -> tuple[list[str], dict]:
    """Parse any supported file and return (text_chunks, meta_info)."""
    ext = os.path.splitext(filename)[1].lower()

    # CSV / Excel
    if ext in ('.csv', '.xlsx', '.xls'):
        df = pd.read_csv(path) if ext == '.csv' else pd.read_excel(path)
        chunks = []
        for i, row in df.iterrows():
            text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            chunks.append(text)
        preview = df.head(3).fillna("").to_dict(orient="records")
        return chunks, {
            "type": "table",
            "rows": len(df),
            "columns": list(df.columns),
            "preview": preview,
        }

    # PDF
    elif ext == '.pdf':
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        full_text = "\n\n".join(text_parts)
        chunks = chunk_text(full_text)
        preview_text = full_text[:300] + "..." if len(full_text) > 300 else full_text
        return chunks, {
            "type": "pdf",
            "rows": len(chunks),
            "columns": ["content"],
            "preview": [{"content": preview_text}],
            "pages": len(text_parts),
        }

    # TXT
    elif ext == '.txt':
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
        chunks = chunk_text(full_text)
        preview_text = full_text[:300] + "..." if len(full_text) > 300 else full_text
        return chunks, {
            "type": "text",
            "rows": len(chunks),
            "columns": ["content"],
            "preview": [{"content": preview_text}],
        }

    # JSON
    elif ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Handle array of objects
        if isinstance(data, list):
            chunks = []
            for i, item in enumerate(data):
                text = json.dumps(item, ensure_ascii=False)
                chunks.append(text)
            preview = data[:3] if len(data) >= 3 else data
            keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
            return chunks, {
                "type": "json",
                "rows": len(data),
                "columns": keys,
                "preview": [{"content": json.dumps(p)} for p in preview],
            }
        else:
            text = json.dumps(data, indent=2, ensure_ascii=False)
            chunks = chunk_text(text)
            return chunks, {
                "type": "json",
                "rows": len(chunks),
                "columns": ["content"],
                "preview": [{"content": text[:300]}],
            }

    # DOCX
    elif ext == '.docx':
        import docx
        doc = docx.Document(path)
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        chunks = chunk_text(full_text)
        preview_text = full_text[:300] + "..." if len(full_text) > 300 else full_text
        return chunks, {
            "type": "docx",
            "rows": len(chunks),
            "columns": ["content"],
            "preview": [{"content": preview_text}],
        }

    raise HTTPException(400, f"Unsupported file type: {ext}")


def ingest_chunks(chunks: list[str], filename: str):
    documents, metadatas, ids = [], [], []
    for i, text in enumerate(chunks):
        if not text.strip():
            continue
        doc_id = f"{filename}__{i}__{uuid.uuid4().hex[:8]}"
        documents.append(text)
        metadatas.append({"source": filename, "row": i})
        ids.append(doc_id)
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[start:start+batch_size],
            metadatas=metadatas[start:start+batch_size],
            ids=ids[start:start+batch_size],
        )
    return len(documents)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    files: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks, file_meta = parse_file(path, file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")

    count = ingest_chunks(chunks, file.filename)
    meta = load_meta()
    meta[file.filename] = file_meta
    save_meta(meta)

    return {"message": f"Ingested {count} chunks from '{file.filename}'", "rows": count, "columns": file_meta.get("columns", [])}


@app.get("/files")
async def list_files():
    return load_meta()


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    meta = load_meta()
    if filename not in meta:
        raise HTTPException(404, "File not found")
    results = collection.get(where={"source": filename})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    del meta[filename]
    save_meta(meta)
    return {"message": f"Deleted '{filename}'"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    total = collection.count()
    if total == 0:
        raise HTTPException(400, "No data ingested yet. Please upload a file first.")
    where = None
    if req.files:
        where = {"source": req.files[0]} if len(req.files) == 1 else {"source": {"$in": req.files}}
    query_params = {"query_texts": [req.question], "n_results": min(req.top_k, total)}
    if where:
        query_params["where"] = where
    results = collection.query(**query_params)
    context_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    context = "\n".join([f"- {chunk}" for chunk in context_chunks])
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the data context below. Be concise and precise.

Data Context:
{context}

Question: {req.question}

Answer:"""
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return QueryResponse(
        answer=response.content[0].text,
        sources=[{"source": s["source"], "row": s["row"]} for s in sources]
    )


@app.get("/stats")
async def stats():
    return {"total_chunks": collection.count()}


@app.delete("/reset")
async def reset():
    global collection
    chroma_client.delete_collection("rag_data")
    collection = chroma_client.get_or_create_collection(
        name="rag_data", embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )
    if os.path.exists(META_FILE):
        os.remove(META_FILE)
    return {"message": "Knowledge base cleared."}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
