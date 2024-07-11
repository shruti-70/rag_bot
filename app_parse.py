from flask import Flask, request, render_template, jsonify
import os
import logging
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

cache = './cache'

os.makedirs(cache, exist_ok=True)

def load_cached_documents():
    documents = []
    for filename in os.listdir(cache):
        if filename.endswith('.txt'):
            with open(os.path.join(cache, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(text=content, doc_id=filename))
    return documents

logger.info("Initializing models and loading documents...")
llm = Ollama(model="llama3", request_timeout=60.0)
embed_model = resolve_embed_model("local:BAAI/bge-m3")

cached_docs = load_cached_documents()
vector_index = VectorStoreIndex.from_documents(cached_docs, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="pdf_extraction",
            description="Tool for extracting insights from cached PDF documents.",
        ),
    ),
]
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
logger.info("Models and documents loaded successfully")

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    full_response = agent.query(user_message)
    
    concise_prompt = f"Summarize the following in one or two sentences: {full_response}"
    concise_answer = llm.complete(concise_prompt).text

    return jsonify({
        'concise_answer': concise_answer,
        'full_response': str(full_response)
    })

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True)