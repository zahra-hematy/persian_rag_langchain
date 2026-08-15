import streamlit as st
from src.llm import LLM
from src.retriever import HybridRetriever
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

    print("1. Loading embedding model...")
    embedding_model = EmbeddingModel().get_embeddings()
    print("2. Embedding loaded.")

    print("3. Loading FAISS...")
    vector_store = VectorStore.load(
        "indexes",
        embedding_model
    )
    print("4. FAISS loaded.")

    print("5. Creating Retriever...")
    retriever = HybridRetriever(vector_store=vector_store,
        k=5,
        final_k=3,
        semantic_weight=0.6,
        keyword_weight=0.4
    )
   
    print("6. Retriever created.")

    print("7. Loading LLM...")
    llm = LLM().get_llm()
    print("8. LLM loaded.")

    print("9. Creating RAGChain...")
    generator = RAGChain(
        llm,
        retriever
    )
    print("10. RAGChain created.")

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
    answer, chunks, has_answer = generator.generate(question)
    st.write(answer)

    if has_answer:

        with st.expander("Retrieved Chunks"):

            for i, chunk in enumerate(chunks, start=1):

                st.markdown(f"### Chunk {i}")

                st.write(
                    f"**Hybrid Score:** "
                    f"`{chunk['hybrid_score']:.4f}`"
                )

                if chunk["semantic_score"] is not None:

                    st.write(
                        f"**FAISS L2:** "
                        f"`{chunk['semantic_score']:.4f}`"
                    )

                if chunk["bm25_score"] is not None:

                    st.write(
                        f"**BM25:** "
                        f"`{chunk['bm25_score']:.4f}`"
                    )

                st.write(chunk["text"])

                st.caption(
                    f"Page: {chunk['page']} | "
                    f"Source: {chunk['source']}"
                )

                st.divider()

   