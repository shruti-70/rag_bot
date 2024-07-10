import os
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import chromadb
from chromadb.utils import embedding_functions

# Specify the directory where ChromaDB will store its data
persist_directory = "./chroma_db"

# Set up ChromaDB with persistent storage
client = chromadb.PersistentClient(path=persist_directory)

# Set up the embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get the collection
collection = client.get_or_create_collection(
    name="pdf_embeddings", 
    embedding_function=embedding_function
)

# Function to read PDF content
def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                try:
                    text += page.extract_text()
                except Exception as e:
                    print(f"Error extracting text from a page in {file_path}: {str(e)}")
                    continue
        return text
    except PdfReadError as e:
        print(f"Error reading PDF {file_path}: {str(e)}")
        return ""
    except Exception as e:
        print(f"Unexpected error reading PDF {file_path}: {str(e)}")
        return ""

# Process PDFs in the /data folder
data_folder = "data" 
for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(data_folder, filename)
        
        # Read PDF content
        pdf_content = read_pdf(file_path)
        
        # Only process if content was successfully extracted
        if pdf_content:
            try:
                # Add to ChromaDB
                collection.add(
                    documents=[pdf_content],
                    metadatas=[{"source": filename}],
                    ids=[filename]
                )
                print(f"Processed and embedded: {filename}")
            except Exception as e:
                print(f"Error embedding {filename}: {str(e)}")
        else:
            print(f"Skipping {filename} due to reading errors")

print("All PDFs have been processed.")

print("\nVerifying stored embeddings:")
stored_items = collection.get()

if stored_items['ids']:
    print(f"Number of documents successfully embedded: {len(stored_items['ids'])}")
    print("Successfully embedded documents:")
    for id, metadata in zip(stored_items['ids'], stored_items['metadatas']):
        print(f"- {id} (Source: {metadata['source']})")
else:
    print("No documents were successfully embedded.")

# You can also check the total count of items in the collection
total_count = collection.count()
print(f"\nTotal number of items in the collection: {total_count}")