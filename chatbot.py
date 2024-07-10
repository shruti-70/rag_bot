import os
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import StorageContext
import chromadb
from chromadb.utils import embedding_functions
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

load_dotenv()

db = chromadb.PersistentClient(path="./chroma_db")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

chroma_collection = db.get_or_create_collection("pdf_embeddings", embedding_function=embedding_function)

# Initialize LLM
llm = Ollama(model="llama3", request_timeout=90.0)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load the existing index
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
    embed_model=embed_model
)
# imp
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

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        result = agent.query(prompt)
        print("Agent Response:")
        print(result)
        
        
        try:

            json_result = json.loads(str(result))
            if isinstance(json_result, dict) and 'response' in json_result:
                concise_answer = json_result['response']
            else:
                concise_answer = str(result)
        except json.JSONDecodeError:
            concise_answer = str(result)
        
        print("\nConcise Answer:")
        print(concise_answer)

        # Save the in output , can get rewritten with every new prompt 
        try:
            os.makedirs("output", exist_ok=True)
            with open(os.path.join("output", "extracted_text.txt"), "w") as f:
                f.write(concise_answer)
            print("Saved extracted text to file: extracted_text.txt")
        except Exception as e:
            logging.error(f"Error saving file: {e}")

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        print("An error occurred while processing your query. Please try again.")

logging.info("execution completed.")