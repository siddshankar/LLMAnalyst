import logging
import xml.etree.ElementTree as ET
from PyPDF2 import PdfReader
import uuid
import agent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def read_pdf_sections(pdf_file: str) -> list:
    """
    Reads a PDF file and splits it into sections.
    Here, each page is treated as a separate section.
    
    Returns:
        sections (list): List of dictionaries with keys 'title' and 'text'.
    """
    sections = []
    try:
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            section_title = f"Page {i+1}"
            sections.append({"title": section_title, "text": text})
        logging.info("Extracted %d sections from %s", len(sections), pdf_file)
    except Exception as e:
        logging.error("Error reading PDF file %s: %s", pdf_file, e)
    return sections

def parse_research_response(response_xml: str) -> list:
    """
    Parses the XML response from the LLM using dedicated tags for axiom key, section reference, and summary text.
    
    Expected XML format:
    <response>
      <summaries>
        <summary>
          <axiom_key>UUID here</axiom_key>
          <section_ref>Section Title</section_ref>
          <text>Summary text...</text>
        </summary>
        ...
      </summaries>
    </response>
    
    Returns:
        summaries (list): List of dictionaries with keys 'axiom_key', 'section', and 'summary'.
    """
    summaries = []
    try:
        root = ET.fromstring(response_xml)
        for summary_elem in root.findall(".//summary"):
            key_elem = summary_elem.find("axiom_key")
            axiom = summary_elem.find("axiom")
            section_elem = summary_elem.find("section_ref")
            text_elem = summary_elem.find("text")

            axiom = axiom.text.strip() if axiom is not None and axiom.text else ""
            axiom_key = key_elem.text.strip() if key_elem is not None and key_elem.text else ""
            section = section_elem.text.strip() if section_elem is not None and section_elem.text else ""
            summary_text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""
            
            summaries.append({"axiom_key": axiom_key, "section": section, "summary": summary_text, "axiom": axiom})
    except Exception as e:
        logging.error("Error parsing research response XML: %s", e)
    return summaries


def iterative_research(question: str, axioms: list, pdf_files: list):
    initial_axioms =[]
    for axiom in axioms:
        initial_axioms.append({"id": str(uuid.uuid4()), "text":axiom})
    llmAgent = agent.LitellmFileSystemAgent()

    """
    Performs iterative research over a set of PDF documents to refine axioms and gather evidence.
    
    Steps:
      1. For each PDF, split the document into sections.
      2. For each section, call the LLM with a prompt that includes:
         - The research question
         - The current axioms
         - The section text (with citation info)
         Expect the LLM to return an XML response with summaries linking evidence to axioms.
      3. Update an evidence mapping (and potentially the axioms list) based on the summaries.
      4. In batches (e.g., 2 axioms at a time), perform a deep dive via a prompt to refine or modify axioms based on the collected evidence.
    
    Returns:
        (axioms, evidence_mapping): The final refined axioms and a mapping of each axiom to its evidence.
    """
    # Initialize mapping: each axiom maps to a list of evidence summaries.
    
    evidence_mapping = {axiom[id]: [] for axiom in initial_axioms}
    
    # Process each PDF file.
    for pdf_file in pdf_files:
        sections = read_pdf_sections(pdf_file)
        for section in sections:
            section_text = section["text"]
            section_title = section["title"]
            axioms_for_prompt = "\n".join([f"{axiom['id']}: {axiom['text']}" for axiom in initial_axioms])       
            prompt = (
                f"Context:\n"
                f"Question: {question}\n"
                f"Axioms:\n{axioms_for_prompt}\n\n"
                f"Research Paper Section ({section_title} from {pdf_file}):\n"
                f"{section_text}\n\n"
                "Task: Analyze this section for evidence that supports or refutes the listed axioms if any are listed, or the overarching question. "
                "For each axiom, provide a concise summary of the relevant evidence from the section. "
                "Include a reference to the section where the evidence is found. "
                "If the section suggests modifications or introduces a new axiom, include that suggestion. \n\n"
                "  <axiom_key> - the unique identifier of the axiom\n"
                "  <axiom> - the new, changed, or existing axiom"
                "  <section_ref> - the section title\n"
                "  <text> - a summary of the evidence\n\n"
                "Return your response in XML format as follows:\n"
                "<response>\n"
                "  <summaries>\n"
                "    <summary>\n"
                "      <axiom_key>the axiom key here if existing axiom</axiom_key>\n"
                "      <axiom> the axiom itself here </axiom>"
                "      <section_ref>Section Title</section_ref>\n"
                "      <text>Summary text...</text>\n"
                "    </summary>\n"
                "    ...\n"
                "  </summaries>\n"
                "</response>\n"
            )

            logging.info("Sending research prompt for section %s of file %s", section_title, pdf_file)
            response_xml = llmAgent.Call_llm(prompt)
            logging.info("Received research response:\n%s", response_xml)
            summaries = parse_research_response(response_xml)
            # Update evidence mapping and update axioms if new ones are suggested.
            for s in summaries:
                axiom_key = s.get("axiom_key", "")
                axiom = s.get("axiom", "")
                if axiom_key:
                    if axiom_key not in evidence_mapping:
                        evidence_mapping[axiom_key] = []
                    foundAxiom = False
                    for a in initial_axioms:
                        if a["id"] == axiom_key:
                            a["text"] == axiom
                            foundAxiom = True
                    if not foundAxiom:
                        initial_axioms.append({"id": str(uuid.uuid4()),"text":axiom})
                    evidence_mapping[axiom_key].append({
                        "pdf": pdf_file,
                        "section": s.get("section", section_title),
                        "summary": s.get("summary", "")
                    })
    

    axioms = iterative_refinement(initial_axioms,evidence_mapping,3,2)
    return axioms, evidence_mapping

def refine_axioms(axioms, evidence_mapping):
    """
    Refines the list of axioms based on the provided evidence.
    Instead of grouping axioms randomly, this function builds a combined evidence summary
    and sends it to the LLM, allowing the LLM to decide which axioms appear to overlap,
    contradict, or are otherwise related.
    
    Each axiom is a dictionary with keys 'id', 'text', and 'iter_count'.
    
    Returns:
        refined_axioms (list): Updated list of axioms as dictionaries with 'id' and 'text'.
    """
    llmAgent = agent.LitellmFileSystemAgent()

    # Build a combined evidence summary for all axioms.
    combined_evidence = ""
    for axiom in axioms:
        combined_evidence += f"Axiom (ID: {axiom['id']}, Text: {axiom['text']}):\n"
        if axiom['id'] in evidence_mapping:
            for ev in evidence_mapping[axiom['id']]:
                combined_evidence += f" - From {ev['pdf']} ({ev['section']}): {ev['summary']}\n"
        combined_evidence += "\n"
    
    # Build a prompt that instructs the LLM to identify groups of related axioms.
    prompt = (
        "Based on the following evidence for the axioms:\n"
        f"{combined_evidence}\n\n"
        "Please analyze the axioms and identify pairs or groups that appear to overlap, contradict, "
        "or are otherwise significantly related. For each identified group, provide refined versions "
        "of the axioms that resolve contradictions or clarify relationships. Use the following XML format:\n\n"
        "<response>\n"
        "  <refinements>\n"
        "    <group>\n"
        "      <axiom key='unique_axiom_key1' original='Original Axiom Text'>Refined Axiom Text</axiom>\n"
        "      <axiom key='unique_axiom_key2' original='Original Axiom Text'>Refined Axiom Text</axiom>\n"
        "    </group>\n"
        "    ... (additional groups if necessary)\n"
        "  </refinements>\n"
        "</response>\n\n"
        "If no overlapping or contradictory axioms are found, return the original axioms unchanged using the same XML format."
    )
    
    logging.info("Sending deep dive refinement prompt for all axioms:\n%s", prompt)
    response_xml = agent.Call_llm(prompt)
    logging.info("Received deep dive refinement response:\n%s", response_xml)
    
    # Parse the XML and update the axioms list.
    refined_axioms = parse_refinement_response(response_xml)
    return refined_axioms

def parse_refinement_response(response_xml: str) -> list:
    """
    Parses the XML response from the LLM for axiom refinements.
    
    Expected XML format:
    <response>
      <refinements>
        <group>
          <axiom key="id1" original="Original Axiom Text">Refined Axiom Text</axiom>
          <axiom key="id2" original="Original Axiom Text">Refined Axiom Text</axiom>
        </group>
        ...
      </refinements>
    </response>
    
    Returns:
       refined_axioms: List of axioms as dictionaries with keys 'id' and 'text'.
    """
    refined_axioms = []
    try:
        root = ET.fromstring(response_xml)
        for group in root.findall(".//group"):
            for axiom_elem in group.findall("axiom"):
                key = axiom_elem.get("key", "").strip()
                original = axiom_elem.get("original", "").strip()
                refined_text = axiom_elem.text.strip() if axiom_elem.text else ""
                # If the refined text is empty, fallback to the original text.
                final_text = refined_text if refined_text else original
                refined_axioms.append({"id": key, "text": final_text})
    except Exception as e:
        logging.error("Error parsing refinement response XML: %s", e)
    return refined_axioms

def iterative_refinement(axioms, evidence_mapping, max_iterations=3, removal_threshold=3):
    """
    Iteratively refines the axioms using the accumulated evidence.
    
    Each axiom is a dictionary with 'id', 'text', and 'iter_count'. In each iteration,
    the LLM is prompted to analyze the evidence and refine overlapping or contradictory axioms.
    After each iteration, each axiom's 'iter_count' is incremented. Axioms that have been iterated
    over a number of times (removal_threshold) are removed.
    
    Args:
        axioms (list): List of axioms as dictionaries with keys 'id', 'text', and optionally 'iter_count'.
        evidence_mapping (dict): Mapping of axiom ids to lists of evidence dictionaries.
        max_iterations (int): Maximum number of refinement iterations.
        removal_threshold (int): Maximum iterations allowed per axiom before removal.
    
    Returns:
        refined_axioms (list): The updated list of axioms.
    """
    # Ensure every axiom has an iteration count.
    for axiom in axioms:
        if 'iter_count' not in axiom:
            axiom['iter_count'] = 0

    for iteration in range(max_iterations):
        logging.info("Starting refinement iteration %d", iteration + 1)
        refined_axioms = refine_axioms(axioms, evidence_mapping)
        new_axioms = []
        # Update iteration counts and decide which axioms to keep.
        for refined in refined_axioms:
            # Find the original axiom with the same id.
            orig = next((a for a in axioms if a['id'] == refined['id']), None)
            if orig:
                refined['iter_count'] = orig.get('iter_count', 0) + 1
            else:
                refined['iter_count'] = 1
            # Optionally, if an axiom has been iterated too many times without change, remove it.
            if refined['iter_count'] < removal_threshold:
                new_axioms.append(refined)
            else:
                logging.info("Removing axiom %s after %d iterations", refined['id'], refined['iter_count'])
        axioms = new_axioms
        # If no axioms remain, exit early.
        if not axioms:
            logging.info("All axioms have been removed after iteration %d", iteration + 1)
            break
    return axioms


# Example usage:
if __name__ == '__main__':
    # Assume agent is already initialized and call_llm is set up.
    question = "How does the evidence in current literature support or contradict the hypothesis that increased exercise improves cardiovascular health?"
    initial_axioms = [
        "Regular exercise improves cardiovascular health",
        "Exercise reduces stress levels"
    ]
    pdf_files = ["paper1.pdf", "paper2.pdf"]  # Replace with actual PDF file paths

    refined_axioms, evidence = iterative_research(question, initial_axioms, pdf_files)
    print("Refined Axioms:")
    for ax in refined_axioms:
        print("-", ax)
    print("\nEvidence Mapping:")
    for axiom, evidences in evidence.items():
        print(f"Axiom: {axiom}")
        for ev in evidences:
            print(f"  From {ev['pdf']} ({ev['section']}): {ev['summary']}")
