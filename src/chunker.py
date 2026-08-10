from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextChunker:
    """
    Splits LangChain Document objects into smaller chunks while
    preserving metadata.
    """

    def __init__(self, chunk_size=800, chunk_overlap=200):

        self.splitter = RecursiveCharacterTextSplitter(

            chunk_size=chunk_size,

            chunk_overlap=chunk_overlap,

            separators=[
                "\n\n",
                "؟",
                "!",
                ".",
                "،",
                "\n",
                " "
            ]
        )

    def chunk(
            self, 
            documents: list[Document]) -> list[Document]:

        return self.splitter.split_documents(
            documents
        )
