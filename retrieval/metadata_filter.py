def filter_docs(docs, key=None, value=None):
    if not key:
        return docs
    return [doc for doc in docs if doc.metadata.get(key) == value]