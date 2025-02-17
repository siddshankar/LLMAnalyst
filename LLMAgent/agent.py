import os
import yaml
from litellm import completion  # LiteLLM’s unified completion API

class LitellmFileSystemAgent:
    """
    An agent that uses LiteLLM to interact with the file system.
    It reads configuration from a YAML file to set the API key, model identifier,
    and folders for file I/O. It can read and write files and modify LaTeX files based on instructions.
    """
    def __init__(self, config_file: str = "config.yaml"):
        # Load configuration from YAML file.
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        llm_config = self.config.get("llm", {})
        self.api_key = llm_config.get("api_key", "")
        self.model = llm_config.get("model", "gemini-2.0-flash")
        self.read_folder = self.config.get("read_folder", ".")
        self.write_folder = self.config.get("write_folder", ".")
        self.api_var = self.config.get("api_var", "GEMINI_API_KEY")

        # Set the API key in the environment if provided.
        if self.api_key:
            # For demonstration, we assume the provider uses OPENAI_API_KEY.
            os.environ[self.api_var] = self.api_key

        # Ensure that the read and write folders exist.
        os.makedirs(self.read_folder, exist_ok=True)
        os.makedirs(self.write_folder, exist_ok=True)
