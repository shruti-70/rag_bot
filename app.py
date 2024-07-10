import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import StorageContext
import chromadb
from chromadb.utils import embedding_functions
import logging
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

load_dotenv()

app = Flask(__name__)

# chroma 
db = chromadb.PersistentClient(path="./chroma_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

chroma_collection = db.get_or_create_collection("pdf_embeddings", embedding_function=embedding_function)

llm = Ollama(model="llama3", request_timeout=60.0)


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
    embed_model=embed_model
)

logging.info(f"Loaded {chroma_collection.count()} embeddings from ChromaDB collection.")

query_engine = index.as_query_engine(llm=llm)


tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="pdf_extraction",
            description="Tool for extracting insights from PDF documents.",
        ),
    ),
]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    max_retries = 3
    timeout_seconds = 60

    for attempt in range(max_retries):
        try:
            result = agent.query(prompt)
            try:
                json_result = json.loads(str(result))
                if isinstance(json_result, dict) and 'response' in json_result:
                    concise_answer = json_result['response']
                else:
                    concise_answer = str(result)
            except json.JSONDecodeError:
                concise_answer = str(result)
            
            return jsonify({
                "full_response": str(result),
                "concise_answer": concise_answer
            })

        except Exception as e:
            logging.error(f"Error processing query on attempt {attempt + 1}: {e}")
            time.sleep(timeout_seconds)

    # use llama 3 at the end
    try:
        result = llm.query(prompt)
        concise_answer = result['choices'][0]['text'] if 'choices' in result and len(result['choices']) > 0 else str(result)
        return jsonify({
            "full_response": str(result),
            "concise_answer": concise_answer
        })
    except Exception as e:
        logging.error(f"Error processing fallback query with llama 3: {e}")
        return jsonify({"error": "An error occurred while processing your query."}), 500

if __name__ == '__main__':
    app.run(debug=True)