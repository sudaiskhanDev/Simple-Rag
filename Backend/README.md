# Document Q&A Backend (Django + LangChain RAG)

A Django-based backend that lets you upload PDF documents, index them into a
Chroma vector database, and ask natural-language questions answered using a
Retrieval-Augmented Generation (RAG) pipeline (Mistral embeddings + Groq LLM).

---

## 🚀 Tech Stack

- **Framework:** Django 5.2
- **Vector Store:** ChromaDB (via `langchain-chroma`)
- **Embeddings:** Mistral AI (`mistral-embed`)
- **LLM:** Groq (`openai/gpt-oss-120b`)
- **PDF Parsing:** PyMuPDF (via `langchain-community`)
- **Text Splitting:** LangChain `TokenTextSplitter`
- **CORS:** `django-cors-headers`
- **Env Management:** `python-dotenv`

---

## 📁 Project Structure

```
Backend/
├── chroma_db/              # Persistent vector store (auto-generated)
├── configure/               # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py / asgi.py
├── documents/                # Main Django app
│   ├── models.py             # Document model
│   ├── views.py               # Upload, index, ask endpoints
│   ├── urls.py
│   └── services/
│       ├── pdf_service.py         # PDF loading (PyMuPDF)
│       ├── chunking_service.py    # Text splitting
│       ├── embedding_service.py   # Mistral embeddings
│       ├── vector_service.py      # Chroma vector store logic
│       ├── retriever_service.py   # Document retrieval (MMR search)
│       └── llm_service.py         # Groq LLM + answer generation
├── media/documents/          # Uploaded PDF files
├── db.sqlite3                 # Django database
├── manage.py
└── test.ipynb                 # Notebook used for prototyping the RAG flow
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Backend
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` isn't available yet, install manually:
```bash
pip install django django-cors-headers python-dotenv
pip install langchain langchain-community langchain-text-splitters
pip install langchain-mistralai langchain-groq langchain-chroma
pip install chromadb pymupdf
```

### 4. Set up environment variables
Create a `.env` file in the `Backend/` root:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run database migrations
```bash
python manage.py migrate
```

### 6. Start the development server
```bash
python manage.py runserver
```

The API will be available at `http://127.0.0.1:8000/`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload/` | Upload a PDF file |
| `GET` | `/documents/` | List all uploaded documents |
| `POST` | `/documents/index/` | Index a document into ChromaDB (`document_id`) |
| `GET` | `/documents/indexed/` | List documents currently indexed in ChromaDB |
| `POST` | `/documents/indexed/delete/` | Remove a document's chunks from ChromaDB (`document_id`) |
| `POST` | `/ask/` | Ask a question (`question`) — answered via RAG |

### Example: Upload a file
```bash
curl -X POST http://127.0.0.1:8000/upload/ \
  -F "file=@/path/to/document.pdf"
```

### Example: Index a document
```bash
curl -X POST http://127.0.0.1:8000/documents/index/ \
  -F "document_id=1"
```

### Example: Ask a question
```bash
curl -X POST http://127.0.0.1:8000/ask/ \
  -F "question=What is the real-world problem discussed?"
```

---

## 🧠 How It Works (RAG Pipeline)

1. **Upload** — PDF is saved via the `Document` model (`media/documents/`).
2. **Load** — `pdf_service.py` uses `PyMuPDFLoader` to extract text.
3. **Chunk** — `chunking_service.py` splits text using `TokenTextSplitter`
   (chunk size: 300 tokens, overlap: 20 tokens).
4. **Embed & Store** — `vector_service.py` embeds chunks with Mistral and
   stores them in a persistent ChromaDB collection (`documents`), along with
   `document_id`, `file_name`, and `chunk_id` metadata.
5. **Retrieve** — `retriever_service.py` performs MMR (Maximal Marginal
   Relevance) search, returning the top 3 most relevant chunks for a query.
6. **Generate Answer** — `llm_service.py` builds a context-grounded prompt
   and sends it to Groq's `openai/gpt-oss-120b` model to produce the final
   answer.

---

## 🔑 Notes

- `DEBUG = True` and `ALLOWED_HOSTS = []` are set for local development only
  — update these before deploying to production.
- CORS is currently restricted to `http://127.0.0.1:5500` in `settings.py`;
  update `CORS_ALLOWED_ORIGINS` / `CSRF_TRUSTED_ORIGINS` to match your
  frontend's origin.
- The `SECRET_KEY` in `settings.py` is a development placeholder — replace
  it with a securely generated key before deploying.
- `chroma_db/` and `db.sqlite3` are generated/populated at runtime; they
  don't need to be committed to version control (add them to `.gitignore`
  if not already there).