from config.settings import RERANKER_MODEL

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None

_reranker_model = None
_reranker_load_failed = False


def rerank(query, docs, top_k=5):
    if not docs:
        return []

    model = _get_reranker_model()
    if model is None:
        return docs[:top_k]

    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)
    ranked_docs = [
        doc
        for _, doc in sorted(
            zip(scores, docs),
            key=lambda item: item[0],
            reverse=True,
        )
    ]
    return ranked_docs[:top_k]


def _get_reranker_model():
    global _reranker_model, _reranker_load_failed

    if _reranker_load_failed:
        return None

    if _reranker_model is None and CrossEncoder is not None:
        try:
            _reranker_model = CrossEncoder(RERANKER_MODEL)
        except Exception:
            _reranker_load_failed = True
            return None

    return _reranker_model
