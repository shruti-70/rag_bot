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

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # time the vector search
            vector_search_start = time.time()
            relevant_docs = query_engine.retrieve(prompt)
            vector_search_time = time.time() - vector_search_start
            logging.info(f"Vector search took {vector_search_time:.2f} seconds")
            
            
            llm_start = time.time()
            result = agent.query(prompt)
            llm_time = time.time() - llm_start
            logging.info(f"LLM response generation took {llm_time:.2f} seconds")
            
            total_time = time.time() - start_time
            logging.info(f"Total query processing took {total_time:.2f} seconds")

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
                "concise_answer": concise_answer,
                "vector_search_time": vector_search_time,
                "llm_time": llm_time,
                "total_time": total_time
            })

        except Exception as e:
            logging.error(f"Error processing query on attempt {attempt + 1}: {e}")
            time.sleep(1)  

    # Fallback to using llama3 directly
    try:
        fallback_start = time.time()
        result = llm.complete(prompt)
        fallback_time = time.time() - fallback_start
        logging.info(f"Fallback query took {fallback_time:.2f} seconds")
        
        concise_answer = result.text if hasattr(result, 'text') else str(result)
        return jsonify({
            "full_response": str(result),
            "concise_answer": concise_answer,
            "fallback_time": fallback_time
        })
    except Exception as e:
        logging.error(f"Error processing fallback query with llama 3: {e}")
        return jsonify({"error": "An error occurred while processing your query."}), 500

