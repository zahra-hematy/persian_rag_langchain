from langchain_community.vectorstores import FAISS


class VectorStore:
    """
    Creates, saves and loads a FAISS vector store.
    """

    @staticmethod
    def create(documents, embedding_model):
        """
        Create a FAISS vector store from documents.
        """

        return FAISS.from_documents(documents, embedding_model)

    @staticmethod
    def save(vector_store, folder_path):
        """
        Save FAISS index locally.
        """

        vector_store.save_local(folder_path)

    @staticmethod
    def load(folder_path, embedding_model):
        """
        Load a saved FAISS index.
        """

        return FAISS.load_local(
            folder_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )