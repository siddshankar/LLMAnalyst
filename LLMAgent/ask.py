import logging
import xml.etree.ElementTree as ET
import re
import agent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_distilled_response(response_xml: str) -> (str, list):
    """
    Parses the distilled XML response into a distilled question and a list of axioms.

    Expected XML format:
    <response>
      <distilled_question>Your distilled, informative question here</distilled_question>
      <axioms>
         <axiom id="unique_axiom_id_1">
             <text>A clear statement of the key concept or relation</text>
             <supporting_evidence>A brief rationale or explanation supporting this axiom</supporting_evidence>
         </axiom>
         <axiom id="unique_axiom_id_2">
             <text>A clear statement of another key concept or relation</text>
             <supporting_evidence>A brief rationale or explanation supporting this axiom</supporting_evidence>
         </axiom>
         ... (additional axioms)
      </axioms>
    </response>

    Returns:
        distilled_question (str): The distilled question.
        axioms (list): A list of dictionaries, each with keys 'id', 'text', and 'supporting_evidence'.
    """
    distilled_question = ""
    axioms = []
    try:
        root = ET.fromstring(response_xml)
        
        # Extract the distilled question.
        dq_elem = root.find("distilled_question")
        if dq_elem is not None and dq_elem.text:
            distilled_question = dq_elem.text.strip()
        else:
            logging.warning("No distilled_question element found in the response.")
        
        # Extract each axiom.
        axioms_elem = root.find("axioms")
        if axioms_elem is not None:
            for axiom_elem in axioms_elem.findall("axiom"):
                # Get the unique ID from the attribute.
                axiom_id = axiom_elem.get("id", "").strip()
                
                # Extract the axiom text.
                text_elem = axiom_elem.find("text")
                axiom_text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""
                
                # Extract the supporting evidence.
                evidence_elem = axiom_elem.find("supporting_evidence")
                supporting_evidence = evidence_elem.text.strip() if evidence_elem is not None and evidence_elem.text else ""
                
                axioms.append({
                    "id": axiom_id,
                    "text": axiom_text,
                    "supporting_evidence": supporting_evidence
                })
        else:
            logging.warning("No axioms element found in the response.")
    
    except Exception as e:
        logging.error("Error parsing distilled response XML: %s", e)
    
    return distilled_question, axioms

def parse_llm_response(response_xml: str):
    """
    Parse the XML response from the LLM.
    Returns a tuple (questions, done) where questions is a list of probing questions,
    and done is a boolean indicating whether the LLM is finished.
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

    questions = []
    questions_node = root.find('questions')
    if questions_node is not None:
        for question_elem in questions_node.findall('question'):
            if question_elem.text:
                questions.append(question_elem.text.strip())

    done_node = root.find('done')
    done = (done_node is not None and done_node.text.strip().lower() == "true")
    logging.info("Parsed %d question(s); LLM done: %s", len(questions), done)
    return questions, done

def ask(llmAgent) -> (str,list):
    """
    Interactive method that:
    1. Prompts the user for a question.
    2. Sends the question to an LLM with a prompt for probing questions in XML.
    3. Iteratively displays probing questions and collects user answers.
    4. Repeats until the LLM indicates it fully understands the question.
    """
    user_question = input("Please ask your question: ")
    conversation_context = f"User Question: {user_question}"
    done = False

    while not done:
        # Construct the prompt for the LLM using the current conversation context.
        prompt = (
            f"Analyze the following conversation and generate probing questions in XML format to understand "
            f"the user's query deeper, or end the conversation by sending true in the done tag:"
            "\n\n{conversation_context}\n\n"
            "Return an XML response in the following format:\n"
            "<response>\n"
            "  <questions>\n"
            "     <question>First probing question?</question>\n"
            "     <question>Second probing question?</question>\n"
            "     ...\n"
            "  </questions>\n"
            "  <done>true|false</done>\n"
            "</response>\n"
        )
        logging.info("Sending prompt to LLM:\n%s", prompt)
        response_xml = (prompt)
        logging.info("Received LLM response:\n%s", response_xml)

        questions, done = parse_llm_response(response_xml)
        if not questions:
            logging.warning("No probing questions returned by LLM. Ending conversation.")
            break

        # Ask each probing question and record the answers.
        for q in questions:
            print(f"\nLLM probing question: {q}")
            answer = input("Your answer: ")
            # Append the question and user's answer to the conversation context.
            conversation_context += f"\nLLM: {q}\nUser: {answer}"

        logging.info("Updated conversation context:\n%s", conversation_context)
    
    print("\nLLM has fully understood your question.")
    logging.info("Final conversation context:\n%s", conversation_context)
    
    logging.info("Prompting LLM to come up with distilled question and axioms")
    distill_prompt = (
    "Based on the following conversation from the ask step:\n\n"
    "=== Conversation Start ===\n"
    f"{conversation_context}"
    "=== Conversation End ===\n\n"
    "Task:\n"
    "1. Distill the core question into a single, succinct, and informative question.\n"
    "2. Identify and list the key concepts and relationships that are needed to answer this question. "
    "For each concept or relationship, define an axiom that is supportable and addresses a part of the question.\n\n"
    "Return your response in XML format using the following structure:\n\n"
    "<response>\n"
    "  <distilled_question>Your distilled, informative question here</distilled_question>\n"
    "  <axioms>\n"
    "     <axiom id=\"unique_axiom_id_1\">\n"
    "         <text>A clear statement of the key concept or relation</text>\n"
    "         <supporting_evidence>A brief rationale or explanation supporting this axiom</supporting_evidence>\n"
    "     </axiom>\n"
    "     <axiom id=\"unique_axiom_id_2\">\n"
    "         <text>A clear statement of another key concept or relation</text>\n"
    "         <supporting_evidence>A brief rationale or explanation supporting this axiom</supporting_evidence>\n"
    "     </axiom>\n"
    "     ... (add additional axioms as needed)\n"
    "  </axioms>\n"
    "</response>\n\n"
    "Make sure that each axiom is supportable and directly addresses a component of the distilled question."
    )
    
    response_xml = llmAgent.Call_llm("user",distill_prompt)
    question,axioms = parse_distilled_response(response_xml)
    return question,axioms

if __name__ == '__main__':
    ask()
