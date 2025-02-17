import logging
import xml.etree.ElementTree as ET
import subprocess
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def call_llm(prompt: str) -> str:
    """
    Placeholder function to simulate calling an LLM API.
    Replace this with your actual API call. 

    Expected XML response for code generation:
    <response>
      <code><![CDATA[
         # your python code here
      ]]></code>
      <done>true|false</done>
    </response>

    And for summary generation:
    <response>
      <summary>Your summary text here</summary>
      <graphs>Your graph instructions here</graphs>
    </response>
    """
    logging.info("LLM Prompt:\n%s", prompt)
    # Simulated response for demonstration purposes.
    # You might vary the simulated output based on the prompt in an actual implementation.
    if "Generate python code" in prompt or "modify the python code" in prompt:
        simulated_response = """
        <response>
          <code><![CDATA[
import pandas as pd
import numpy as np

def analyze_data(csv_file):
    # Read CSV data
    df = pd.read_csv(csv_file)
    # Conduct basic statistical analysis:
    # (e.g., computing descriptive statistics)
    results = df.describe().to_dict()
    # Output the analysis results
    print("Statistical Analysis Results:")
    print(results)

if __name__ == '__main__':
    analyze_data("dummy.csv")
        ]]></code>
          <done>false</done>
        </response>
        """
    else:
        # Simulated summary response.
        simulated_response = """
        <response>
          <summary><![CDATA[
The analysis indicates that the dataset provides moderate support for the provided axioms.
Key statistics suggest trends consistent with the hypothesis, although further data cleaning may be required.
          ]]></summary>
          <graphs><![CDATA[
Bar charts comparing means across groups and scatter plots to show correlation trends.
          ]]></graphs>
        </response>
        """
    return simulated_response

def parse_code_response(response_xml: str) -> (str, bool):
    """
    Parse the XML response from the LLM for code generation.
    Returns a tuple (code, done) where code is the python code string,
    and done is a boolean indicating if the analysis is complete.
    """
    try:
        root = ET.fromstring(response_xml)
    except ET.ParseError as e:
        logging.error("Error parsing XML: %s", e)
        return "", False

    code_elem = root.find('code')
    code_text = ""
    if code_elem is not None and code_elem.text:
        code_text = code_elem.text.strip()

    done_elem = root.find('done')
    done = (done_elem is not None and done_elem.text.strip().lower() == "true")
    logging.info("Parsed code; LLM done flag: %s", done)
    return code_text, done

def parse_summary_response(response_xml: str) -> (str, str):
    """
    Parse the XML response from the LLM for summary and graph instructions.
    Returns a tuple (summary, graphs) as strings.
    """
    try:
        root = ET.fromstring(response_xml)
    except ET.ParseError as e:
        logging.error("Error parsing summary XML: %s", e)
        return "", ""
    
    summary_elem = root.find('summary')
    graphs_elem = root.find('graphs')
    summary_text = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
    graphs_text = graphs_elem.text.strip() if graphs_elem is not None and graphs_elem.text else ""
    return summary_text, graphs_text

def iterative_statistical_analysis(question: str, axioms: list, csv_file: str):
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
                f"Using the following information:\n{conversation_context}\n\n"
                "Generate python code that performs statistical analysis on the dataset provided in the CSV file. "
                "The code should analyze the dataset to assess support for the axioms or the question, using appropriate "
                "statistical techniques. Return your response in XML format with a <code> tag containing the python code, "
                "and a <done> tag indicating if the analysis is complete (true/false)."
            )
        else:
            prompt = (
                f"The previously generated code was executed and produced the following output:\n{output}\n\n"
                "Please modify the python code as necessary to improve the statistical analysis so that it more clearly addresses "
                "the question and the provided axioms (or suggests a new axiom if needed). Return your response in XML format "
                "with a <code> tag for the updated code and a <done> tag indicating if the analysis is complete (true/false)."
            )
        
        logging.info("Iteration %d - Sending prompt to LLM for code generation.", iteration)
        response_xml = call_llm(prompt)
        logging.info("Iteration %d - Received LLM response:\n%s", iteration, response_xml)
        
        new_code, done = parse_code_response(response_xml)
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
        
        # Run the generated python code and capture its output
        try:
            result = subprocess.run(["python", file_name], capture_output=True, text=True, check=True)
            output = result.stdout
            logging.info("Iteration %d - Code execution output:\n%s", iteration, output)
        except subprocess.CalledProcessError as e:
            output = e.stdout + "\nError:\n" + e.stderr
            logging.error("Iteration %d - Error during code execution:\n%s", iteration, output)
    
    # After iterations, ask LLM to generate a summary and graph instructions
    summary_prompt = (
        f"Based on the final analysis output below:\n{output}\n\n"
        "Generate a summary of the statistical analysis results and provide instructions for any graphs that would help "
        "visualize the findings. Return your response in XML format with a <summary> tag for the text summary and a "
        "<graphs> tag for the graph instructions."
    )
    logging.info("Sending final prompt to LLM for summary and graph generation.")
    summary_response_xml = call_llm(summary_prompt)
    logging.info("Received summary response from LLM:\n%s", summary_response_xml)
    
    summary, graphs = parse_summary_response(summary_response_xml)
    logging.info("Final summary: %s", summary)
    logging.info("Graph instructions: %s", graphs)
    
    return summary, graphs

if __name__ == '__main__':
    # Example usage with dummy parameters (ensure that 'data.csv' exists in the same folder)
    question = "Does the dataset support the hypothesis that increased exercise improves health outcomes?"
    axioms = [
        "Regular exercise improves cardiovascular health",
        "Exercise reduces stress levels"
    ]
    csv_file = "data.csv"  # Replace with your actual CSV file
    
    final_summary, graph_instructions = iterative_statistical_analysis(question, axioms, csv_file)
    
    print("\nFinal Summary:\n", final_summary)
    print("\nGraph Instructions:\n", graph_instructions)
