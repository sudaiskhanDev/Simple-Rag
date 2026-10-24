from langchain_mistralai import MistralAIEmbeddings


def get_embedding_model():
    """
    Create and return the Mistral embedding model.
    """

    embedding_model = MistralAIEmbeddings(
        model="mistral-embed"
    )

    return embedding_model