import logging
import uuid
import ask
import analysis
import write
import research
import agent

# Assume the following methods are defined elsewhere in your codebase:
# - ask(): Prompts the user to enter a question and returns it.
# - iterative_research(question, initial_axioms, pdf_files): Processes PDFs and returns (refined_axioms, evidence_mapping, research_summaries)
# - iterative_statistical_analysis(question, axioms, csv_file): Runs analysis and returns (analysis_summary, graph_instructions)
# - write_paper(question, evidence_mapping, axioms, research_excerpts, max_iterations): Iteratively writes the paper and returns the final paper text.
# - call_llm(): Your function to call the LLM.
#
# For demonstration, dummy implementations or sample data may be used.

def run_research_pipeline():
    """
    Connects the overall research process:
      1. Asks the research question.
      2. Performs research on provided documents to refine axioms and gather evidence.
      3. Runs statistical analysis using the refined axioms and a CSV dataset.
      4. Writes the final research paper using all collected information.
    
    Returns:
        final_paper (str): The complete research paper.
    """
    logging.info("Starting research pipeline...")
    
    # Step 1: Ask the question.
    question,initial_axioms  = ask.ask()  # assume this function prompts the user and returns the question.
    logging.info("Research Question: %s", question)
    
    # Step 2: Research Step.
    # Define initial axioms (using unique IDs) and list of PDF file paths.
    

    pdf_files,csv_file = get_pdf_files()  # Replace with actual PDF file paths.
    
    # Run the research method. Assume it returns:
    #   refined_axioms: updated list of axioms,
    #   evidence_mapping: a dict mapping axiom IDs to lists of evidence dictionaries,
    #   research_summaries: a list of summary dictionaries.
    logging.info("Starting research on provided documents...")
    refined_axioms, evidence_mapping, research_summaries = research.iterative_research(question, initial_axioms, pdf_files)
    logging.info("Refined Axioms: %s", refined_axioms)
    
    # (Optional) Combine summaries into a block text if needed for later steps.
    summaries_text = ""
    for s in research_summaries:
        summaries_text += f"Axiom {s['axiom_key']} (Section: {s['section']}): {s['summary']}\n"
    
    # Step 3: Analysis Step.
    # Assume we have a CSV file (e.g., "data.csv") that will be analyzed using the refined axioms.
    logging.info("Running statistical analysis using refined axioms...")
    analysis_summary, graph_instructions = analysis.iterative_statistical_analysis(question, refined_axioms, csv_file)
    logging.info("Analysis Summary: %s", analysis_summary)
    analysis_evidence = {
        "pdf": "Statistical Analysis Report",
        "section": "Analysis",
        "summary": analysis_summary + "\nGraph Instructions: " + graph_instructions
    }
    # Add a special analysis axiom so the evidence mapping can reference it.
    analysis_axiom = {"id": "analysis", "text": "Statistical analysis results", "iter_count": 0}
    refined_axioms.append(analysis_axiom)
    evidence_mapping["analysis"] = [analysis_evidence]
    
    logging.info("Composing the final research paper...")
    final_paper = write.write_paper(question, evidence_mapping, refined_axioms, max_iterations=5)
    
    
    logging.info("Research pipeline complete.")
    return final_paper

def get_pdf_files():
        folder = "resources"
        # list all files in the folder and filter for .pdf files (case-insensitive)
        pdf_files = [f for f in os.listdir(folder) 
                    if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.pdf')]
        csv = ''
        for f in os.listdir(folder):
             if os.path.isfile(os.path.join(folder,f)) and f.lower().endswith('.csv'):
                  csv = f
        return pdf_files, csv 
# Example usage:
if __name__ == '__main__':
    # Configure logging to display INFO-level messages.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Run the complete research pipeline.
    paper = run_research_pipeline()
    print("Final Research Paper:\n")
    print(paper)
