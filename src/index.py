from pathlib import Path
import time
from loader import PDFLoader
from cleaner import TextCleaner
from chunker import TextChunker
from embedding import EmbeddingModel
from vector_store import VectorStore


def main():

    start_time = time.time()

    print("Loading PDF documents...")

    loader = PDFLoader()

    documents = loader.load(
        "data"
    )

    print(
        f"Loaded {len(documents)} pages."
    )

    # -----------------------------
    # Clean documents
    # -----------------------------

    cleaner = TextCleaner()

    documents = cleaner.clean(
        documents
    )

    print("Documents cleaned.")

    # -----------------------------
    # Chunk documents
    # -----------------------------

    chunker = TextChunker()

    chunks = chunker.chunk(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # -----------------------------
    # Embedding model
    # -----------------------------

    embedding_model = EmbeddingModel().get_embeddings()

    # -----------------------------
    # Build FAISS
    # -----------------------------

    print("Creating FAISS index...")

    vector_store = VectorStore.create(
        chunks,
        embedding_model
    )

    # -----------------------------
    # Save index
    # -----------------------------

    Path("indexes").mkdir(
        exist_ok=True
    )

    VectorStore.save(
        vector_store,
        "indexes"
    )

    print("FAISS index saved.")

    print(
        f"Finished in {time.time() - start_time:.2f} seconds."
    )


if __name__ == "__main__":

    main()