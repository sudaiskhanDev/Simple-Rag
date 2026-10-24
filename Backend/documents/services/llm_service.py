from langchain_groq import ChatGroq


def get_llm():
    """
    Create and return the Groq LLM.
    """

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3
    )

    return llm




def generate_answer(question, retrieved_documents):
    """
    Generate an answer using the retrieved document context.
    """

    context = "\n\n".join(
        document.page_content

        for document in retrieved_documents
    )


    prompt = f"""
You are a document question-answering assistant in a Retrieval-Augmented Generation (RAG) system.

Your task is to answer the user's question using ONLY the information provided in the retrieved context.

Follow these instructions carefully:

1. Use only the retrieved context as your source of information.
   - Do not use your own general knowledge.
   - Do not make assumptions.
   - Do not invent or add facts that are not explicitly supported by the context.

2. Identify the information in the context that is relevant to the user's question.
   - Ignore information that is unrelated to the question.
   - Focus only on the parts of the context that help answer the question.

3. When multiple retrieved passages contain the same or very similar information:
   - Remove unnecessary repetition.
   - Combine overlapping information into one clear statement.
   - Preserve any important details that appear in one passage but not another.

4. When the retrieved context contains incomplete or conflicting information:
   - Do not guess or resolve the conflict using outside knowledge.
   - Answer only with what can be supported by the provided context.
   - Clearly state when the available context is insufficient to give a definite answer.

5. Provide a clear, direct, and concise answer.
   - Answer the question directly.
   - Do not unnecessarily describe the retrieval process.
   - Do not mention that you are an AI unless necessary.
   - Do not include information that is unrelated to the user's question.

6. If the context does not contain enough information to answer the question:
   - Say that the provided documents do not contain enough relevant information to answer the question.
   - Do not generate an answer based on assumptions or outside knowledge.

Retrieved Context:
{context}

User Question:
{question}

Final Answer:
"""

    llm = get_llm()

    response = llm.invoke(prompt)


    return response.content