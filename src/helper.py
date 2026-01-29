from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings



def load_pdf_files_from_directory(data):
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    
    """Filter documents to only include page_content and source metadata."""
    
    
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )       
        
    return minimal_docs


# split texts into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
   )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks



def download_embeddings():
    model_name="BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name
    )
    return embeddings

embedding = download_embeddings()