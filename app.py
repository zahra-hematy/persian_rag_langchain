import streamlit as st
from src.llm import LLM
from src.retriever import Retriever
from src.rag_chain import RAGChain
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore


st.set_page_config(

    page_title="Persian RAG",

    page_icon="📄",

    layout="wide"
)


@st.cache_resource
def load_rag():

    # -----------------------------
    # Embedding Model
    # -----------------------------

    embedding_model = EmbeddingModel().get_embeddings()

    # -----------------------------
    # Load FAISS
    # -----------------------------

    vector_store = VectorStore.load(

        "indexes",

        embedding_model
    )

    # -----------------------------
    # Retriever
    # -----------------------------

    retriever = Retriever(

        vector_store
    ).get_retriever()

    # -----------------------------
    # LLM
    # -----------------------------

    llm = LLM().get_llm()

    # -----------------------------
    # Generator
    # -----------------------------

    generator = RAGChain(

        llm,

        retriever
    )

    return generator


generator = load_rag()

# ----------------------------------
# UI
# ----------------------------------

st.title("📄 Persian RAG (LangChain)")

question = st.text_input(

    "Question",

    placeholder="Ask a question..."
)

if st.button(

    "Ask",

    use_container_width=True
):

    if not question.strip():

        st.warning("Please enter a question.")

        st.stop()

    # with st.spinner("Searching..."):
    answer, chunks = generator.generate(
        question
    )

    st.write(answer)
    with st.expander("Retrieved Chunks"):

        for i, chunk in enumerate(chunks, start=1):

            st.markdown(f"### Chunk {i}")

            st.write(chunk["text"])

            st.caption(
                f'Page: {chunk["page"]} | Source: {chunk["source"]}'
            )

            st.divider()