from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import Document


@csrf_exempt
def upload_file(request):

    if request.method == "POST":

        file = request.FILES.get("file")

        if not file:
            return JsonResponse({
                "error": "No file provided"
            }, status=400)

        document = Document.objects.create(
            file=file
        )

        return JsonResponse({
            "message": "File uploaded successfully",
            "file_id": document.id,
            "file_name": document.file.name,
        })

    return JsonResponse({
        "message": "Upload a file using POST request"
    })








def list_documents(request):

    if request.method == "GET":

        documents = Document.objects.all().order_by("-uploaded_at")

        data = []

        for document in documents:
            data.append({
                "file_id": document.id,
                "file_name": document.file.name,
                "uploaded_at": document.uploaded_at,
            })

        return JsonResponse({
            "documents": data
        })

    return JsonResponse({
        "error": "Only GET request is allowed"
    }, status=405)








from .services.pdf_service import load_pdf
from .services.chunking_service import split_documents
from .services.vector_service import add_documents,is_document_indexed,get_indexed_documents,delete_document

@csrf_exempt   
def index_document(request):

    if request.method != "POST":
        return JsonResponse({
            "error": "Only POST request is allowed"
        }, status=405)

    document_id = request.POST.get("document_id")

    if not document_id:
        return JsonResponse({
            "error": "document_id is required"
        }, status=400)

    try:
        document_id = int(document_id)

    except ValueError:
        return JsonResponse({
            "error": "document_id must be an integer"
        }, status=400)

    try:
        document = Document.objects.get(id=document_id)

    except Document.DoesNotExist:
        return JsonResponse({
            "error": "Document not found"
        }, status=404)

    # Check if document is already indexed
    if is_document_indexed(document.id):
        return JsonResponse({
            "message": "Document is already indexed",
            "document_id": document.id,
            "file_name": document.file.name
        })

    # 1. Load PDF
    docs = load_pdf(document.file.path)

    # 2. Split into chunks
    chunks = split_documents(docs)

    # 3. Add chunks and embeddings to ChromaDB
    add_documents(
        chunks=chunks,
        document_id=document.id,
        file_name=document.file.name
    )

    return JsonResponse({
        "message": "Document indexed successfully",
        "document_id": document.id,
        "file_name": document.file.name,
        "chunks": len(chunks)
    })


def indexed_documents(request):

    if request.method != "GET":
        return JsonResponse({
            "error": "Only GET request is allowed"
        }, status=405)

    documents = get_indexed_documents()

    return JsonResponse({
        "documents": documents
    })





@csrf_exempt
def delete_indexed_document(request):

    if request.method != "POST":
        return JsonResponse({
            "error": "Only POST request is allowed"
        }, status=405)

    document_id = request.POST.get("document_id")

    if not document_id:
        return JsonResponse({
            "error": "document_id is required"
        }, status=400)

    try:
        document_id = int(document_id)

    except ValueError:
        return JsonResponse({
            "error": "document_id must be an integer"
        }, status=400)

    deleted = delete_document(document_id)

    if not deleted:
        return JsonResponse({
            "message": "Document is not indexed in ChromaDB",
            "document_id": document_id
        }, status=404)

    return JsonResponse({
        "message": "Document removed from ChromaDB successfully",
        "document_id": document_id
    })





from .services.retriever_service import retrieve_documents
from .services.llm_service import generate_answer

@csrf_exempt
def ask_question(request):

    if request.method != "POST":
        return JsonResponse({
            "error": "Only POST request is allowed"
        }, status=405)

    question = request.POST.get("question")

    if not question:
        return JsonResponse({
            "error": "Question is required"
        }, status=400)

    # 1. Retrieve relevant chunks from ChromaDB
    retrieved_documents = retrieve_documents(question)

    if not retrieved_documents:
        return JsonResponse({
            "message": "No relevant information found",
            "answer": None
        })

    # 2. Generate answer using retrieved context
    answer = generate_answer(
        question=question,
        retrieved_documents=retrieved_documents
    )

    return JsonResponse({
        "question": question,
        "answer": answer
    })