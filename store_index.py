from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
load_dotenv()
Pinecone_api_key = os.environ.get('Pinecone_api_key')
Pinecone_api_env = os.environ.get('Pinecone_api_env')
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone

# Print a masked version of the API key for verification
print(f"Using API key: {Pinecone_api_key[:4]}...{Pinecone_api_key[-4:]}")

# Initialize Pinecone
pc = Pinecone(api_key=Pinecone_api_key)

index_name = "medimate"  # Your chosen index name

# Check if the index already exists
if index_name not in pc.list_indexes().names():
    # Create the index if it doesn't exist
    pc.create_index(
        name=index_name,
        dimension=384,  # dimension of the all-MiniLM-L6-v2 embeddings
        metric='cosine'
    )

# Get the index
index = pc.Index(index_name)

# Manually create embeddings for each text chunk and upsert to Pinecone
for i, chunk in enumerate(text_chunks):
    embedding = embeddings.embed_query(chunk.page_content)
    try:
        index.upsert(vectors=[(f"id_{i}", embedding, {"text": chunk.page_content})])
    except Exception as e:
        print(f"Error upserting vector {i}: {str(e)}")
        break

# Use LangchainPinecone to interact with the index
docsearch = LangchainPinecone(
    index,
    embedding_function=embeddings.embed_query,
    text_key="text"
)

    