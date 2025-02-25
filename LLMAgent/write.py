import logging
import xml.etree.ElementTree as ET
import agent

def parse_write_response(response_xml: str) -> (dict, dict, bool):
    """
    Parses the XML response from the LLM for paper writing.
    
    Expected XML format:
    <response>
      <updated_skeleton>
        <section name="Abstract">... updated abstract ...</section>
        <section name="Introduction">... updated introduction ...</section>
        <section name="Methods">... updated methods ...</section>
        <section name="Results">... updated results ...</section>
        <section name="Discussion">... updated discussion ...</section>
        <section name="Conclusion">... updated conclusion ...</section>
      </updated_skeleton>
      <!-- Optional deep dive instruction -->
      <deep_dive section="Results" axiom_key="123e4567-e89b-12d3-a456-426614174000">
         Please provide more details on the statistical analysis in this section.
      </deep_dive>
      <done>true/false</done>
    </response>
    
    Returns:
        updated_skeleton (dict): Mapping of section names to their updated content.
        deep_dive (dict): Mapping of section names to deep dive info, where each value is a dict with keys:
                          'axiom_key' (may be empty) and 'text' (the deep dive instruction).
        done (bool): True if the LLM indicates the paper is complete.
    """
    updated_skeleton = {}
    deep_dive = {}
    done = False
    try:
        root = ET.fromstring(response_xml)
        skeleton_node = root.find("updated_skeleton")
        if skeleton_node is not None:
            for section_elem in skeleton_node.findall("section"):
                section_name = section_elem.get("name", "").strip()
                content = section_elem.text.strip() if section_elem.text else ""
                updated_skeleton[section_name] = content
        # Look for any deep dive tags.
        for dd in root.findall("deep_dive"):
            sec = dd.get("section", "").strip()
            axiom_key = dd.get("axiom_key", "").strip() if dd.get("axiom_key") else ""
            dd_text = dd.text.strip() if dd.text else ""
            deep_dive[sec] = {"axiom_key": axiom_key, "text": dd_text}
        # Check if done.
        done_elem = root.find("done")
        if done_elem is not None and done_elem.text.strip().lower() == "true":
            done = True
    except Exception as e:
        logging.error("Error parsing write response XML: %s", e)
    return updated_skeleton, deep_dive, done

def write_paper(question: str, evidence_mapping: dict, axioms: list, research_excerpts: str, max_iterations: int = 5) -> str:
    """
    Iteratively writes a research paper by first generating a skeleton and then refining it.
    
    The paper is constructed using:
      - The original research question.
      - Evidence mapping: A dictionary mapping axiom IDs to lists of evidence dictionaries (with keys 'pdf', 'section', 'summary').
      - Defined axioms (key concepts and relations).
      - Research excerpts: Text from the research papers used in the research.
    
    In each iteration, the current paper skeleton is sent to the LLM along with the above evidence.
    If the LLM requests a deep dive into a particular section and (optionally) a specific axiom,
    the code retrieves the relevant evidence from the evidence mapping for inclusion.
    
    Args:
        question (str): The original research question.
        evidence_mapping (dict): Mapping of axiom IDs to lists of evidence dictionaries.
        axioms (list): A list of axioms. Each axiom is a dict with keys 'id' and 'text'.
        research_excerpts (str): Combined text excerpts from the research papers.
        max_iterations (int): Maximum iterations allowed for refinement.
    
    Returns:
        final_paper (str): The fully composed research paper.
    """
    # Initialize a default skeleton.
    skeleton = {
        "Abstract": "Placeholder abstract.",
        "Introduction": "Placeholder introduction.",
        "Methods": "Placeholder methods.",
        "Results": "Placeholder results.",
        "Discussion": "Placeholder discussion.",
        "Conclusion": "Placeholder conclusion."
    }
    
    iteration = 0
    done = False
    # Placeholder for additional deep dive evidence to include in the prompt.
    deep_dive_evidence = ""
    
    # Build an "Evidence Mapping" text block from the evidence_mapping.
    evidence_text = ""
    for axiom in axioms:
        evidences = evidence_mapping.get(axiom['id'], [])
        if evidences:
            evidence_text += f"Axiom {axiom['id']} ({axiom['text']}):\n"
            for ev in evidences:
                evidence_text += f" - From {ev['pdf']} (Section: {ev['section']}): {ev['summary']}\n"
            evidence_text += "\n"
    
    while not done and iteration < max_iterations:
        iteration += 1
        
        axioms_text = ""
        for ax in axioms:
            axioms_text += f"{ax['id']}: {ax['text']}\n"
        
        # Build the current skeleton text.
        skeleton_text = ""
        for sec, content in skeleton.items():
            skeleton_text += f"Section: {sec}\n{content}\n\n"
        
        # Build the prompt, including research excerpts and evidence mapping.
        prompt = (
            "You are tasked with writing a cohesive research paper that answers the following question:\n"
            f"{question}\n\n"
            "The paper must include proper citations and be supported by evidence from the provided evidence mapping, key concepts (axioms), "
            "and research excerpts (text from original papers used in the research).\n\n"
            "Evidence Mapping (for citations):\n"
            f"{evidence_text}\n\n"
            "Key Concepts and Axioms:\n"
            f"{axioms_text}\n\n"
            "Research Excerpts (from original papers):\n"
            f"{research_excerpts}\n\n"
        )
        
        # If there is deep dive evidence from a previous iteration, include it.
        if deep_dive_evidence:
            prompt += (
                "Deep Dive Evidence from previous request:\n"
                f"{deep_dive_evidence}\n\n"
            )
        
        prompt += (
            "The current skeleton of the paper is as follows:\n"
            f"{skeleton_text}\n\n"
            "Task:\n"
            "1. Refine and update the paper skeleton. Fill in details for each section, especially the Abstract, which should summarize the entire paper.\n"
            "2. Incorporate evidence, citations, and relevant text excerpts into the appropriate sections.\n"
            "3. If additional detail is needed for a specific section, include a <deep_dive section='SectionName' axiom_key='OptionalAxiomID'> tag with instructions.\n"
            "4. Return your response in XML format with the following structure:\n\n"
            "<response>\n"
            "  <updated_skeleton>\n"
            "    <section name='Abstract'>... updated abstract ...</section>\n"
            "    <section name='Introduction'>... updated introduction ...</section>\n"
            "    <section name='Methods'>... updated methods ...</section>\n"
            "    <section name='Results'>... updated results ...</section>\n"
            "    <section name='Discussion'>... updated discussion ...</section>\n"
            "    <section name='Conclusion'>... updated conclusion ...</section>\n"
            "  </updated_skeleton>\n"
            "  <deep_dive section='SectionName' axiom_key='OptionalAxiomID'>Additional details or request for deep dive on that section (include relevant research text if needed)</deep_dive>\n"
            "  <done>true/false</done>\n"
            "</response>\n\n"
            "Ensure that the final paper is cohesive, answers the question, and includes proper citations."
        )
        llmAgent = agent.LitellmFileSystemAgent()
        logging.info("Iteration %d: Sending write prompt to LLM.", iteration)
        response_xml = llmAgent.Call_llm("user",prompt)
        logging.info("Iteration %d: Received write response:\n%s", iteration, response_xml)
        
        updated_skeleton, deep_dive, done = parse_write_response(response_xml)
        
        # Update our current skeleton with the updated content.
        for section, content in updated_skeleton.items():
            skeleton[section] = content
        
        # Check if a deep dive was requested.
        deep_dive_evidence = ""
        if deep_dive:
            for sec, info in deep_dive.items():
                requested_axiom = info.get("axiom_key", "")
                dd_instruction = info.get("text", "")
                # Gather evidence from the evidence mapping for the specified section.
                evidence_for_dd = ""
                if requested_axiom:
                    evidences = evidence_mapping.get(requested_axiom, [])
                    for ev in evidences:
                        if ev["section"].lower() == sec.lower():
                            evidence_for_dd += f"Axiom {requested_axiom}: From {ev['pdf']} (Section: {ev['section']}): {ev['summary']}\n"
                else:
                    # If no axiom is specified, include all evidence for that section.
                    for key, evidences in evidence_mapping.items():
                        for ev in evidences:
                            if ev["section"].lower() == sec.lower():
                                evidence_for_dd += f"Axiom {key}: From {ev['pdf']} (Section: {ev['section']}): {ev['summary']}\n"
                
                deep_dive_evidence += f"Deep Dive Request for Section {sec}: {dd_instruction}\nEvidence:\n{evidence_for_dd}\n\n"
                # Optionally, append the deep dive details directly to the current skeleton section.
                skeleton[sec] += f"\n\nDeep Dive Details:\n{dd_instruction}\nEvidence:\n{evidence_for_dd}"
        
        if done:
            logging.info("Final paper completed after iteration %d.", iteration)
        else:
            logging.info("Paper not complete after iteration %d. Continuing refinement.", iteration)
    
    # Compose the final paper text.
    final_paper = ""
    for section, content in skeleton.items():
        final_paper += f"{section}\n{'-' * len(section)}\n{content}\n\n"
    
    return final_paper

# Example usage:
if __name__ == '__main__':
    # Dummy inputs for demonstration purposes.
    question = "How does increased physical activity contribute to improved cardiovascular health, and what are the underlying mechanisms?"
    
    # Example evidence mapping (typically generated from previous research steps)
    evidence_mapping = {
        "123e4567-e89b-12d3-a456-426614174000": [
            {
                "pdf": "paper1.pdf",
                "section": "Introduction",
                "summary": "Studies indicate a significant reduction in heart disease risk with regular exercise. [Doe et al., 2020]"
            }
        ],
        "223e4567-e89b-12d3-a456-426614174001": [
            {
                "pdf": "paper2.pdf",
                "section": "Discussion",
                "summary": "Evidence shows improved arterial function and reduced inflammation due to physical activity. [Smith et al., 2019]"
            }
        ]
    }
    
    # Example axioms (key concepts)
    axioms = [
        {"id": "123e4567-e89b-12d3-a456-426614174000", "text": "Regular exercise improves cardiovascular function"},
        {"id": "223e4567-e89b-12d3-a456-426614174001", "text": "Physical activity reduces arterial inflammation"}
    ]
    
    # Example research excerpts from the original papers.
    research_excerpts = (
        "From Doe et al., 2020 (Introduction): 'Regular physical activity is associated with a lower risk of coronary heart disease...'\n"
        "From Smith et al., 2019 (Methods): 'Our study shows that exercise improves arterial elasticity and reduces inflammatory markers...'\n"
    )
    
    final_paper = write_paper(question, evidence_mapping, axioms, research_excerpts, max_iterations=5)
    print("Final Research Paper:\n")
    print(final_paper)
