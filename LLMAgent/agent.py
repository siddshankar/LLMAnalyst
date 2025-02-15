import os
from litellm import completion  # LiteLLM’s unified completion API

class LitellmFileSystemAgent:
    """
    An agent that uses LiteLLM to interact with the file system.
    It can read and write files and specifically modify LaTeX files according to given instructions.
    """
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the agent with the specified LiteLLM model.
        
        Args:
            model (str): The model identifier to use.
        """
        self.model = model

    def read_file(self, file_path: str) -> str:
        """
        Read the contents of a file.
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            str: Contents of the file.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def write_file(self, file_path: str, content: str):
        """
        Write content to a file.
        
        Args:
            file_path (str): Path to the file.
            content (str): Content to be written.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def edit_latex_file(self, file_path: str, instructions: str) -> str:
        """
        Edit a LaTeX file using LiteLLM based on provided instructions.
        
        Args:
            file_path (str): Path to the LaTeX file.
            instructions (str): Modification instructions.
            
        Returns:
            str: The updated LaTeX content.
        """
        original_content = self.read_file(file_path)
        prompt = (
            "You are an expert LaTeX editor. Given the following LaTeX document:\n"
            f"{original_content}\n\n"
            "Please apply the following modifications:\n"
            f"{instructions}\n\n"
            "Return only the updated LaTeX document."
        )
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        # Extract the updated content from the OpenAI-compatible response
        updated_content = response["choices"][0]["message"]["content"]
        self.write_file(file_path, updated_content)
        return updated_content
