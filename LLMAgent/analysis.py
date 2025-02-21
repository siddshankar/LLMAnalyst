import logging
import xml.etree.ElementTree as ET
import subprocess
import os
import re
import agent
import kagglehub

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_code_response(response_xml: str) -> (str, bool,list):
    """
    Parse the XML response from the LLM for code generation.
    Returns a tuple (code, done) where code is the python code string,
    and done is a boolean indicating if the analysis is complete.
    """
    match = re.search(r'<(\w+)[^>]*>.*?</\1>', response_xml, re.DOTALL)
    if match:
        first_element_str = match.group(0)
        print(first_element_str)
        # Now you can parse this string if needed
        try:
            root = ET.fromstring(first_element_str)
            # Further processing with 'root'
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
    else:
        print("No valid XML element found.")


    code_elem = root.find('code')
    code_text = ""
    if code_elem is not None and code_elem.text:
        code_text = code_elem.text.strip()
    req_elem = None
    try:
        req_elem = root.find('requirements').text
    except:
        print("no requirements.txt, no need to run")

    done_elem = root.find('done')
    done = (done_elem is not None and done_elem.text.strip().lower() == "true")

    logging.info("Parsed code; LLM done flag: %s", done)

    return code_text, done, req_elem

def parse_summary_response(response_xml: str) -> (str, str):
    """
    Parse the XML response from the LLM for summary and graph instructions.
    Returns a tuple (summary, graphs) as strings.
    """
    match = re.search(r'<(\w+)[^>]*>.*?</\1>', response_xml, re.DOTALL)
    if match:
        first_element_str = match.group(0)
        print(first_element_str)
        # Now you can parse this string if needed
        try:
            root = ET.fromstring(first_element_str)
            # Further processing with 'root'
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
    else:
        print("No valid XML element found.")

    summary_elem = root.find('summary')
    graphs_elem = root.find('graphs')
    summary_text = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
    graphs_text = graphs_elem.text.strip() if graphs_elem is not None and graphs_elem.text else ""
    return summary_text, graphs_text

def iterative_statistical_analysis(llmAgent: agent.LitellmFileSystemAgent,question: str, axioms: list, csv_file: str, method: str):
    """
    Iteratively generates and refines python code to conduct statistical analysis on a given dataset.
    The analysis is guided by the provided question and axioms.
    
    Steps:
      1. Create a prompt (with the question, axioms, and CSV file info) for the LLM to generate python code.
      2. Save the generated code to a .py file and run it.
      3. Feed the execution output back to the LLM to refine the code.
      4. Repeat until the LLM indicates completion.
      5. Finally, prompt the LLM to generate a summary and graph instructions.
    """
    conversation_context = f"Question: {question}\nAxioms: {axioms}\nCSV File: {csv_file}"
    current_code = ""
    iteration = 0
    max_iterations = 5
    done = False
    output = ""
    file_name = "generated_analysis.py"
    
    while not done and iteration < max_iterations:
        iteration += 1
        if iteration == 1:
            prompt = (
        f"Given the following context:\n"
        f"Question: {question}\n"
        f"Axioms: {axioms}\n"
        f"CSV File: {csv_file}\n\n"
        f"Your task is to generate Python code that performs a thorough statistical analysis using {method} on the dataset provided in the CSV file. "
        "The code should evaluate whether the dataset supports the provided axioms or answers the question by using appropriate "
        "statistical techniques (such as hypothesis testing, descriptive statistics, etc.). In your analysis, consider "
        "handling potential issues such as missing values, outliers, or data transformations.\n\n"
        "There will be several iterations of creating the code and modifying it, please get important information like the size and columns"
        "of the dataset in the first iteration, and debug and polish the code over the next few iterations."
        "Return your response in XML format. The XML should include a <code> tag containing the Python code and a <done> tag indicating "
        "if the analysis is complete (true/false). In addition, please include any dependencies in the <requirements> tag separated by new line as shown \n\n"
        "For example, your response might look like:\n"
        "<response>\n"
        "  <code><![CDATA[\n"
        "import pandas as pd\n"
        "import numpy as np\n\n"
        "def analyze_data(csv_file):\n"
        "    df = pd.read_csv(csv_file)\n"
        f"    # Perform descriptive statistics using {method} \n"
        "    stats = df.describe()\n"
        "    # Optionally add hypothesis testing or regression analysis here\n"
        "    print('Analysis Results:', stats)\n\n"
        "if __name__ == '__main__':\n"
        "    analyze_data('your_file.csv')\n"
        "]]></code>\n"
        "  <done>false</done>\n"
        "<requirements>xgb\nscipy\nsklearn\n</requirements>\n"
        "</response>\n"
    )
        else:
            prompt = (
                f"Original Context:\n"
                f"Question: {question}\n"
                f"Axioms: {axioms}\n"
                f"CSV File: {csv_file}\n\n"
                "Previously generated Python code:\n"
                f"{current_code}\n\n"
                "The above code was executed and produced the following output:\n"
                f"{output}\n\n"
                "This output reflects the current state of the statistical analysis. Based on the original context and the execution output, "
                "please refine the Python code to better address the question and the provided axioms. Consider adjustments such as improving data handling, "
                "applying alternative statistical methods, or incorporating additional insights (e.g., suggesting new axioms if warranted by the data).\n\n"
                "Return your updated code in XML format with a <code> tag for the Python code and a <done> tag indicating whether the analysis is now complete (true/false).\n\n"
                "For example, your response might be:\n"
                "<response>\n"
                "  <code><![CDATA[\n"
                "# Updated code with improved analysis...\n"
                "]]></code>\n"
                "  <done>false</done>\n"
                "</response>\n"
            )

        
        logging.info("Iteration %d - Sending prompt to LLM for code generation.", iteration)
        response = llmAgent.Call_llm("user", prompt)
        logging.info("Iteration %d - Received LLM response:\n%s", iteration, response)
        
        new_code, done, req_elem = parse_code_response(response)
        if not new_code:
            logging.error("Iteration %d - No code returned by LLM. Exiting loop.", iteration)
            break
        current_code = new_code
        
        # Save the generated code to a .py file
        try:
            with open(file_name, "w") as f:
                f.write(current_code)
            logging.info("Iteration %d - Saved generated code to %s", iteration, file_name)
        except Exception as e:
            logging.error("Iteration %d - Error writing code to file: %s", iteration, e)
            break
        
        if req_elem != None:
            try:
                with open("requirements.txt", "w") as f:
                    f.write(req_elem)
                logging.info("Iteration %d - Saved requirements.txt to %s", iteration, "requirements.txt")
            except Exception as e:
                logging.error("Iteration %d - Error writing reqs to file: %s", iteration, e)
                break

            try:
                # Install packages from requirements.txt
                subprocess.run(["pip", 'install', '-r', 'requirements.txt'])

            except subprocess.CalledProcessError as e:
                output = output = e.stdout + "\nError:\n" + e.stderr
                logging.error("Iteration %d - Error during requirements installation:\n%s", iteration, output)
                done = False


        # Run the generated python code and capture its output
        try:
            result = subprocess.run(["python", file_name], capture_output=True, text=True, check=True)
            output = result.stdout
            logging.info("Iteration %d - Code execution output:\n%s", iteration, output)
        except subprocess.CalledProcessError as e:
            output = e.stdout + "\nError:\n" + e.stderr
            logging.error("Iteration %d - Error during code execution:\n%s", iteration, output)
            done = False
    
    # After iterations, ask LLM to generate a summary and graph instructions
    summary_prompt = (
    "Previous Context:"
    f"Question: {question}\n"
    f"Axioms: {axioms}\n"
    f"Based on the final output of the statistical analysis:\n{output}\n\n"
    "Please provide a comprehensive summary of the analysis, including key statistical findings, observed trends, and any conclusions "
    "regarding the support for the provided axioms or the answer to the question.\n\n"
    "Return your response in XML format with a <summary> tag containing the text summary and a <graphs> tag for the graph instructions.\n\n"
    "For example, your response might be:\n"
    "<response>\n"
    "  <summary><![CDATA[\n"
    "The analysis reveals a significant positive correlation between variable X and outcome Y, supporting the hypothesis that ...\n"
    "]]></summary>\n"
    "  <graphs><![CDATA[\n"
    "Generate a scatter plot of X versus Y with a regression line overlay and a histogram of the residuals.\n"
    "]]></graphs>\n"
    "</response>\n"
)
    logging.info("Sending final prompt to LLM for summary and graph generation.")
    summary_response_xml = llmAgent.Call_llm("user", summary_prompt)
    logging.info("Received summary response from LLM:\n%s", summary_response_xml)
    
    summary, graphs = parse_summary_response(summary_response_xml)
    logging.info("Final summary: %s", summary)
    logging.info("Graph instructions: %s", graphs)
    
    return summary, graphs

if __name__ == '__main__':
    # Example usage with dummy parameters (ensure that 'data.csv' exists in the same folder)
    question = "How do various factors affect house prices?"
    axioms = [
        "Square footage is correlated with number of bedrooms and therefore house price",
        "Location is also important as it can increase or decrease crime rate and house size and therefore prices"
    ]

    path = kagglehub.dataset_download("sukhmandeepsinghbrar/housing-price-dataset")

    csv_file = path  # Replace with your actual CSV file

    llmAgent = agent.LitellmFileSystemAgent()

    final_summary, graph_instructions = iterative_statistical_analysis(llmAgent ,question, axioms, csv_file, "XGBoost")
    print("\nFinal Summary:\n", final_summary)
    print("\nGraph Instructions:\n", graph_instructions)
