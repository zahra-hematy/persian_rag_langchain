from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from rank_bm25 import BM25Okapi

from typing import List
import re


class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever using:
    - FAISS semantic search
    - BM25 keyword search

    Final ranking is based on a weighted combination
    of normalized FAISS and BM25 scores.
    """

    vector_store: object

    k: int = 6
    final_k: int = 5

    semantic_weight: float = 0.6
    keyword_weight: float = 0.4

    documents: List[Document] = []

    bm25: object = None

    def model_post_init(self, __context):

        documents = list(
            self.vector_store.docstore._dict.values()
        )

        self.documents = documents

        tokenized_documents = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    @staticmethod
    def _tokenize(text: str):

        text = text.lower()

        text = re.sub(
            r"[^\w\u0600-\u06FF\-]+",
            " ",
            text
        )

        return text.split()

    def _get_relevant_documents(
        self,
        query: str
    ) -> List[Document]:

        # =====================================
        # 1. FAISS
        # =====================================

        faiss_results = (
            self.vector_store
            .similarity_search_with_score(
                query,
                k=self.k
            )
        )

        faiss_scores = {}

        for document, score in faiss_results:

            doc_id = id(document)

            faiss_scores[doc_id] = {
                "document": document,
                "score": float(score)
            }

        # =====================================
        # 2. BM25
        # =====================================

        query_tokens = self._tokenize(query)

        bm25_raw_scores = self.bm25.get_scores(
            query_tokens
        )

        bm25_ranked = sorted(
            enumerate(bm25_raw_scores),
            key=lambda x: x[1],
            reverse=True
        )[:self.k]

        bm25_scores = {}

        for index, score in bm25_ranked:

            document = self.documents[index]

            doc_id = id(document)

            bm25_scores[doc_id] = {
                "document": document,
                "score": float(score)
            }

        # =====================================
        # 3. Collect candidates
        # =====================================

        candidate_ids = set()

        candidate_ids.update(
            faiss_scores.keys()
        )

        candidate_ids.update(
            bm25_scores.keys()
        )

        # =====================================
        # 4. Normalize FAISS
        # =====================================

        faiss_values = [
            item["score"]
            for item in faiss_scores.values()
        ]

        if faiss_values:

            faiss_min = min(faiss_values)
            faiss_max = max(faiss_values)

        else:

            faiss_min = 0
            faiss_max = 1

        # =====================================
        # 5. Normalize BM25
        # =====================================

        bm25_values = [
            item["score"]
            for item in bm25_scores.values()
        ]

        if bm25_values:

            bm25_min = min(bm25_values)
            bm25_max = max(bm25_values)

        else:

            bm25_min = 0
            bm25_max = 1

        # =====================================
        # 6. Hybrid score
        # =====================================

        ranked_documents = []

        for doc_id in candidate_ids:

            document = None

            semantic_score = None
            keyword_score = None

            # -----------------------------
            # FAISS score
            # -----------------------------

            if doc_id in faiss_scores:

                document = (
                    faiss_scores[doc_id]["document"]
                )

                semantic_score = (
                    faiss_scores[doc_id]["score"]
                )

                if faiss_max != faiss_min:

                    semantic_normalized = (
                        faiss_max - semantic_score
                    ) / (
                        faiss_max - faiss_min
                    )

                else:

                    semantic_normalized = 1.0

            else:

                semantic_normalized = 0.0

            # -----------------------------
            # BM25 score
            # -----------------------------

            if doc_id in bm25_scores:

                document = (
                    bm25_scores[doc_id]["document"]
                )

                keyword_score = (
                    bm25_scores[doc_id]["score"]
                )

                if bm25_max != bm25_min:

                    keyword_normalized = (
                        keyword_score - bm25_min
                    ) / (
                        bm25_max - bm25_min
                    )

                else:

                    keyword_normalized = 1.0

            else:

                keyword_normalized = 0.0

            # -----------------------------
            # Hybrid
            # -----------------------------

            hybrid_score = (

                self.semantic_weight
                * semantic_normalized

                +

                self.keyword_weight
                * keyword_normalized
            )

            # -----------------------------
            # Store metadata
            # -----------------------------

            document.metadata[
                "semantic_score"
            ] = semantic_score

            document.metadata[
                "bm25_score"
            ] = keyword_score

            document.metadata[
                "hybrid_score"
            ] = float(hybrid_score)

            document.metadata[
                "semantic_weight"
            ] = self.semantic_weight

            document.metadata[
                "keyword_weight"
            ] = self.keyword_weight

            ranked_documents.append(
                (
                    document,
                    hybrid_score
                )
            )

        # =====================================
        # 7. Sort
        # =====================================

        ranked_documents.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # =====================================
        # 8. Final results
        # =====================================

        return [
            document
            for document, score
            in ranked_documents[
                :self.final_k
            ]
        ]