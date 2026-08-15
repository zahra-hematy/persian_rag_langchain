from langchain.chains import create_retrieval_chain

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)
from langchain_core.prompts import ChatPromptTemplate


class RAGChain:
    """
    Creates a Retrieval-Augmented Generation chain.
    """

    def __init__(self, llm, retriever):

        prompt = ChatPromptTemplate.from_template(
            """
        You are a Persian document assistant.

        Use ONLY the provided context.

        If the answer exists:

        - Answer only in Persian.
        - Summarize the information.
        - Do not copy the document word by word.

        If the answer does not exist reply exactly:

        اطلاعات کافی در اسناد موجود نیست.

        Context:

        {context}

        Question:

        {input}
        """
        )

        document_chain = create_stuff_documents_chain(llm, prompt)

        self.chain = create_retrieval_chain(retriever, document_chain)

    def generate(self, question):

        response = self.chain.invoke(
            {
                "input": question
            }
        )
        answer = response["answer"]

        # ----------------------------------
        # Check whether the answer exists
        # ----------------------------------

        no_answer_text = "اطلاعات کافی در اسناد موجود نیست."

        has_answer = (no_answer_text not in answer.strip())

        # ----------------------------------
        # Retrieved chunks
        # ----------------------------------

        chunks = []

        for doc in response["context"]:

            chunks.append(
                {
                    "text": doc.page_content,
                    "page": doc.metadata["page"],
                    "source": doc.metadata["source"],
                    "semantic_score": doc.metadata.get("semantic_score"),
                    "bm25_score": doc.metadata.get("bm25_score"),
                    "hybrid_score": doc.metadata.get("hybrid_score")
                }
            )
        return response["answer"], chunks, has_answer