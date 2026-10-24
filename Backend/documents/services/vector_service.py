from pathlib import Path

from django.conf import settings
from langchain_chroma import Chroma

from .embedding_service import get_embedding_model


CHROMA_PATH = str(Path(settings.BASE_DIR) / "chroma_db")


def get_vector_store():
    """
    Get the persistent Chroma vector store.
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name="documents",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    return vector_store


def add_documents(chunks, document_id, file_name):
    """
    Add document chunks to ChromaDB with metadata.
    """

    vector_store = get_vector_store()

    for index, chunk in enumerate(chunks):

        chunk.metadata["document_id"] = document_id
        chunk.metadata["file_name"] = file_name
        chunk.metadata["chunk_id"] = index

    vector_store.add_documents(chunks)

    return vector_store


def is_document_indexed(document_id):
    """
    Check whether a document is already indexed in ChromaDB.
    """

    vector_store = get_vector_store()

    results = vector_store.get(
        where={
            "document_id": document_id
        },
        limit=1
    )

    return len(results["ids"]) > 0


def get_indexed_documents():
    """
    Get the list of documents currently indexed in ChromaDB.
    """

    vector_store = get_vector_store()

    results = vector_store.get()

    documents = {}

    for metadata in results["metadatas"]:

        if not metadata:
            continue

        document_id = metadata.get("document_id")
        file_name = metadata.get("file_name")

        if document_id is not None:
            documents[document_id] = file_name

    return [
        {
            "document_id": document_id,
            "file_name": file_name
        }
        for document_id, file_name in documents.items()
    ]


def delete_document(document_id):
    """
    Delete all chunks belonging to a document from ChromaDB.
    """

    vector_store = get_vector_store()

    results = vector_store.get(
        where={
            "document_id": document_id
        }
    )

    ids = results["ids"]

    if not ids:
        return False

    vector_store.delete(ids=ids)

    return True