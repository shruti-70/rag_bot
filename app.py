import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from chromadb.utils import embedding_functions
import logging
import time
import concurrent.futures
import functools

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

app = Flask(__name__)

# Chroma DB setup
db = chromadb.PersistentClient(path="./chroma_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_collection = db.get_or_create_collection("pdf_embeddings", embedding_function=embedding_function)

llm = Ollama(model="llama3", request_timeout=60.0)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
    service_context=service_context
)

embedding_count = chroma_collection.count()
logging.info(f"Loaded {embedding_count} embeddings from ChromaDB collection.")

query_engine = index.as_query_engine(
    similarity_top_k=3,
    streaming=False
)

def timeout(timeout_duration):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_duration)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout_duration} seconds")
        return wrapper
    return decorator

@timeout(60)
def query_with_timeout(prompt):
    return query_engine.query(prompt)

@app.route('/inspect_db', methods=['GET'])
def inspect_db():
    try:
        items = chroma_collection.get()
        return jsonify({
            "total_items": len(items['ids']),
            "sample_items": [{
                "id": id,
                "metadata": metadata,
                "content": content[:100] + "..."
            } for id, metadata, content in zip(items['ids'][:10], items['metadatas'][:10], items['documents'][:10])]
        })
    except Exception as e:
        logging.error(f"Error inspecting ChromaDB: {e}", exc_info=True)
        return jsonify({"error": "An error occurred while inspecting the database."}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    for attempt in range(3):
        try:
            start_time = time.time()
            response = query_with_timeout(prompt)
            total_time = time.time() - start_time
            logging.info(f"Query processing successful on attempt {attempt + 1}, took {total_time:.2f} seconds")
            return jsonify({
                "response": str(response),
                "total_time": total_time
            })
        except TimeoutError:
            logging.warning(f"Timeout on attempt {attempt + 1}")
        except Exception as e:
            logging.error(f"Error processing query on attempt {attempt + 1}: {e}", exc_info=True)

    # Fallback to using llama3 directly
    try:
        fallback_start = time.time()
        result = llm.complete(prompt)
        fallback_time = time.time() - fallback_start
        
        logging.info(f"Fallback query took {fallback_time:.2f} seconds")
        
        return jsonify({
            "response": result.text if hasattr(result, 'text') else str(result),
            "fallback_time": fallback_time
        })
    except Exception as e:
        logging.error(f"Error processing fallback query: {e}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your query."}), 500

if __name__ == '__main__':
    app.run(debug=True)