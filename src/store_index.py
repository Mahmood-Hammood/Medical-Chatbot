# load environment variables (.env file)
from dotenv import load_dotenv
import os
load_dotenv()
from pinecone import Pinecone
from pinecone import ServerlessSpec
from src.helper import load_pdf_files_from_directory, filter_to_minimal_docs, text_split, download_embeddings
from langchain_pinecone import PineconeVectorStore




extracted_texts = load_pdf_files_from_directory("../data")
filter_data = filter_to_minimal_docs(extracted_texts)
text_chunks = text_split(filter_data)

embedding = download_embeddings()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot-index"

if not pc.has_index(index_name):
    index = pc.create_index(
        name=index_name,
        dimension=384,  # dimension of the embedding vector  or len(embedding.embed_query("test"))
        metric="cosine",  # similarity metric
        spec=ServerlessSpec(
           cloud="aws",
           region="us-east-1",
        )
    )
    
index = pc.Index(index_name)




db = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name,
)   