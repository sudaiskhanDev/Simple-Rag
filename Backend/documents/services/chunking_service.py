from langchain_text_splitters import TokenTextSplitter


def split_documents(documents):
    """
    Split loaded documents into smaller chunks.
    """

    splitter = TokenTextSplitter(
        chunk_size=300,
        chunk_overlap=20
    )


    chunks = splitter.split_documents(documents)


    return chunks