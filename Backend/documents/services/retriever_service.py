from .vector_service import get_vector_store


def get_retriever():
    """
    Create and return the Chroma retriever.
    """

    vector_store = get_vector_store()


    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":9
        }
    )

    return retriever



def retrieve_documents(question):
    """
    Retrieve relevant document chunks for a question.
    """

    retriever = get_retriever()

    results= retriever.invoke(question)

    return results

