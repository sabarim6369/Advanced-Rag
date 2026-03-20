from langchain_community.document_loaders import PyPDFLoader

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_multiple_documents(file_paths):
    documents = []
    for file_path in file_paths:
        documents.extend(load_documents(file_path))
    return documents
