from src.helper import download_hugging_face_embeddings
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv 
from src.prompt import prompt as prompt_template
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
app = Flask(__name__)
print(sys.path)
import llama_cpp
print(llama_cpp.__file__)

load_dotenv()
PINECONE_API_KEY = os.environ.get('Pinecone_api_key')
PINECONE_API_ENV = os.environ.get('Pinecone_api_env')
embeddings = download_hugging_face_embeddings()
index_name = "medimate"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)
docsearch = LangchainPinecone(index, embeddings.embed_query, "text")
# Assuming embeddings is used to embed the user input

prompt_template = prompt_template()
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}
from langchain.llms import LlamaCpp
model_path = os.path.abspath("model/llama-2-7b-chat.Q5_K_M.gguf")
llm = LlamaCpp(model_path=model_path, n_ctx=2048, n_threads=8)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
@app.route("/")
def index_get():
    return render_template("chat.html")
import traceback  # Add this to import error handling

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        print(f"User input: {msg}")
        embedded_query = embeddings.embed_query(msg)
        result = index.query(vector=embedded_query, top_k=2)
        qa_result = qa({"query": msg})
        print(f"LLM Response: {qa_result['result']}")
        return jsonify(qa_result["result"])
    except Exception as e:
        # Print the error traceback to help debug the issue
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()  # This will output the full stack trace to the console
        
        # Return an error message to the frontend
        return jsonify({"error": "Something went wrong on the server"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True,use_reloader=True)


