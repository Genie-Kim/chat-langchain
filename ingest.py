"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader


from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def ingest_docs():
    """Get documents from web pages."""
    # loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/")
    # loader = UnstructuredPDFLoader("mydocs/gpt-4.pdf")
    # loader = UnstructuredPDFLoader("https://arxiv.org/pdf/2302.03803.pdf",mode="elements")
    loader = UnstructuredPDFLoader("mydocs/VisualChatGPT.pdf", mode="elements")

    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    
    ingest_docs()
