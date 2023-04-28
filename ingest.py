"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader


from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def ingest_docs(pdf_text):
    """Get documents from web pages."""
    
    # loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/")
    loader = UnstructuredPDFLoader(pdf_text, mode="elements")   

    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    
    # parse the pdf file's name
    pdf_name = pdf_text.split("/")[-1]
    pdf_name = pdf_name.split(".")[0] 
    print(pdf_name)
    
    # Save vectorstore
    with open(f"vector_drive/{pdf_name}.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    # get params from command line
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str, default=None)
    args = parser.parse_args()
    
    if args.pdf is not None:
        ingest_docs(f'downloads/{args.pdf}')
    else:
        for filename in os.listdir("downloads"):
            if filename.endswith(".pdf"):
            # Load data from the pickle file
                with open(os.path.join("downloads", filename), 'rb') as f:
                    ingest_docs(f'downloads/{filename}')
    
