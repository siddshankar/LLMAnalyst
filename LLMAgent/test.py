import os
import tempfile
import unittest
import yaml
from unittest.mock import patch

from agent import LitellmFileSystemAgent

class TestLitellmFileSystemAgent(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for read and write folders.
        self.temp_read_folder = tempfile.mkdtemp()
        self.temp_write_folder = tempfile.mkdtemp()

        # Create a temporary config file.
        self.temp_config_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml", encoding="utf-8")
        config_data = {
            "llm": {
                "api_key": "AIzaSyC9prhgpFUhbDFCI6PbRTFjBYFioFwj2Yc",
                "model": "gemini-2.0-flash",
                "api_var": "GEMINI_API_KEY"
            },
            "read_folder": self.temp_read_folder,
            "write_folder": self.temp_write_folder
        }
        yaml.dump(config_data, self.temp_config_file)
        self.temp_config_file.close()

        self.agent = LitellmFileSystemAgent(config_file=self.temp_config_file.name)

    def tearDown(self):
        # Clean up temporary directories and config file.
        import shutil
        shutil.rmtree(self.temp_read_folder)
        shutil.rmtree(self.temp_write_folder)
        os.remove(self.temp_config_file.name)

    def test_read_file(self):
        # Create a temporary file in the read folder.
        file_name = "test.txt"
        file_path = os.path.join(self.temp_read_folder, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Test content")
        
        content = self.agent.read_file(file_name)
        self.assertEqual(content, "Test content")

    def test_read_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.agent.read_file("non_existent.txt")

    def test_write_file(self):
        file_name = "output.txt"
        self.agent.write_file(file_name, "New content")
        file_path = os.path.join(self.temp_write_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "New content")

    @patch("agent.completion")
    def test_edit_latex_file(self, mock_completion):
        original_latex = r"""
\documentclass{article}
\begin{document}
Hello World!
\end{document}
"""
        instructions = "Replace 'Hello World!' with 'Hello, LaTeX World!'"
        updated_latex = r"""
\documentclass{article}
\begin{document}
Hello, LaTeX World!
\end{document}
"""
        mock_completion.return_value = {
            "choices": [
                {"message": {"content": updated_latex}}
            ]
        }
        # Write the original LaTeX file into the read folder.
        file_name = "document.tex"
        file_path = os.path.join(self.temp_read_folder, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(original_latex)
        
        result = self.agent.edit_latex_file(file_name, instructions)
        self.assertEqual(result, updated_latex)
        # Verify that the file was updated in the write folder.
        output_path = os.path.join(self.temp_write_folder, file_name)
        with open(output_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        self.assertEqual(file_content, updated_latex)
        # Confirm that the prompt passed to completion() contains the instructions.
        args, _ = mock_completion.call_args
        prompt_content = args[0]["messages"][0]["content"]
        self.assertIn("Replace 'Hello World!'", prompt_content)

if __name__ == "__main__":
    unittest.main()
