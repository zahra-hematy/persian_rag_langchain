class Retriever:
    """
    Wraps LangChain Retriever.
    """

    def __init__(
        self,
        vector_store,
        k=3
    ):

        self.retriever = vector_store.as_retriever(

            search_type="similarity",

            search_kwargs={
                "k": k
            }
        )

    def get_retriever(self):

        return self.retriever