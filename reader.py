# Ensure code_reader is defined before tools
from llama_index.core.tools import FunctionTool
import os

def reader_func(file_name):
    path = os.path.join("data", file_name)
    try:
        with open(path, "r") as f:
            content = f.read()
            return {"file_content": content}
    except Exception as e:
        return {"error": str(e)}

reader = FunctionTool.from_defaults(
    fn=reader_func,
    name="reader",
    description="""this tool can read the contents of files and answer questions derived from the text
    . Use this when you need to read the contents of a file""",
)

# Define context and code_parser_template after tools are fully initialized
context = """Purpose: To extract information from pdf file and give insightful answers. """

code_parser_template = """Parse the response from a previous LLM into a description and a string of valid code, 
                            also come up with a valid filename this could be saved as that doesnt contain special characters. 
                            Here is the response: {response}. You should parse this in the following JSON Format: """

# Ensure 'output' directory exists before writing files
if not os.path.exists("output"):
    os.makedirs("output")

# Rest of your script remains unchanged but ensure proper JSON parsing and error handling.
