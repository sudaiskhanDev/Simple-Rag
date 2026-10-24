from django.urls import path

from .views import (
    upload_file,
    list_documents,
    index_document,
    indexed_documents,
    delete_indexed_document,
    ask_question
)


urlpatterns = [
    path("upload/", upload_file, name="upload_file"),
    path("documents/", list_documents, name="list_documents"),
    path("documents/index/", index_document, name="index_document"),
    path("documents/indexed/", indexed_documents, name="indexed_documents"),
    path(
        "documents/indexed/delete/",
        delete_indexed_document,
        name="delete_indexed_document"
    ),
    path("ask/", ask_question, name="ask_question")
]