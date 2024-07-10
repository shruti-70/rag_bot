from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from dotenv import load_dotenv
import os
import ast
from prompts import code_parser_template

load_dotenv()


llm = Ollama(model="llama3", request_timeout=60.0)
parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()


embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

# ReAct agent
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="pdf_extraction",
            description="Tool for extracting insights from PDF documents.",
        ),
    ),
    # pdf reader comes here
]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# Define output parsing pipeline
class ExtractionOutput(BaseModel):
    text_content: str

parser = PydanticOutputParser(ExtractionOutput)
json_prompt_str = parser.format(code_parser_template)  #
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])


while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            break
        except Exception as e:
            retries += 1
            print(f"Error occurred, retry #{retries}:", e)

    if retries >= 3:
        print("Unable to process request, try again...")
        continue

    
    print("Text extracted:")
    print(cleaned_json["text_content"])

    try:
        
        with open(os.path.join("output", "extracted_text.txt"), "w") as f:
            f.write(cleaned_json["text_content"])
        print("Saved extracted text to file: extracted_text.txt")
    except Exception as e:
        print(f"Error saving file: {e}")
