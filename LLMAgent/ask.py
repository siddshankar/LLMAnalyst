import logging
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def call_llm(prompt: str) -> str:
    """
    Placeholder function to simulate calling an LLM API.
    Replace the simulated response with an actual API call.
    
    The expected XML response format is:
    <response>
      <questions>
         <question>First probing question?</question>
         <question>Second probing question?</question>
         ...
      </questions>
      <done>true|false</done>
    </response>
    """
    logging.info("Calling LLM with prompt: %s", prompt)
    # Simulated response for demonstration purposes:
    simulated_response = """
    <response>
      <questions>
         <question>What is the main subject of your question?</question>
         <question>Can you provide more context or details?</question>
      </questions>
      <done>false</done>
    </response>
    """
    return simulated_response

def parse_llm_response(response_xml: str):
    """
    Parse the XML response from the LLM.
    Returns a tuple (questions, done) where questions is a list of probing questions,
    and done is a boolean indicating whether the LLM is finished.
    """
    try:
        root = ET.fromstring(response_xml)
    except ET.ParseError as e:
        logging.error("Error parsing XML: %s", e)
        return [], False

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

def ask():
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
        response_xml = call_llm(prompt)
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
    return conversation_context

if __name__ == '__main__':
    ask()
