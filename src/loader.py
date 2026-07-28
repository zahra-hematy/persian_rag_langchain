from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader


class PDFLoader:
    """
    Loads PDF documents and returns LangChain Document objects.
    """

    def load(self, folder_path: str):

        documents = []

        pdf_folder = Path(folder_path)

        pdf_files = pdf_folder.glob("*.pdf")

        for pdf_file in pdf_files:

            loader = PyMuPDFLoader(
                str(pdf_file)
            )

            docs = loader.load()

            documents.extend(
                docs
            )

        return documents