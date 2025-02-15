import os
import tempfile
import unittest
from unittest.mock import patch

from agent import LitellmFileSystemAgent

class TestLitellmFileSystemAgent(unittest.TestCase):
    def setUp(self):
        self.agent = LitellmFileSystemAgent()

    def test_read_file(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as temp_file:
            temp_file.write("Test content")
            temp_file_path = temp_file.name

        try:
            content = self.agent.read_file(temp_file_path)
            self.assertEqual(content, "Test content")
        finally:
            os.remove(temp_file_path)

    def test_read_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.agent.read_file("non_existent_file.txt")

    def test_write_file(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as temp_file:
            temp_file_path = temp_file.name

        try:
            self.agent.write_file(temp_file_path, "New content")
            with open(temp_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, "New content")
        finally:
            os.remove(temp_file_path)

    @patch("agent.completion")
    def test_edit_latex_file(self, mock_completion):
        # Set up test LaTeX content and modification instructions.
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
        # Configure the mock to return a fake response matching LiteLLM’s output schema.
        mock_completion.return_value = {
            "choices": [
                {"message": {"content": updated_latex}}
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(original_latex)
            temp_file_path = temp_file.name

        try:
            result = self.agent.edit_latex_file(temp_file_path, instructions)
            self.assertEqual(result, updated_latex)
            # Verify that the file content was updated.
            with open(temp_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            self.assertEqual(file_content, updated_latex)
            # Confirm that the prompt passed to the completion call contains the instruction.
            args, _ = mock_completion.call_args
            prompt_content = args[0]["messages"][0]["content"]
            self.assertIn("Replace 'Hello World!'", prompt_content)
        finally:
            os.remove(temp_file_path)

if __name__ == "__main__":
    unittest.main()
