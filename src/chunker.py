from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    Splits LangChain Document objects into smaller chunks while
    preserving metadata.
    """

    def __init__(self, chunk_size=500, chunk_overlap=100):

        self.splitter = RecursiveCharacterTextSplitter(

            chunk_size=chunk_size,

            chunk_overlap=chunk_overlap,

            separators=[
                "\n\n",
                "\n",
                "؟",
                "!",
                ".",
                " "
            ]
        )

    def chunk(self, documents: List[Document]) -> List[Document]:

        return self.splitter.split_documents(
            documents
        )