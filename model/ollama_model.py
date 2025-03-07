import logging
import json
import sys
from typing import Any, Dict, List, Optional

from langchain.llms.base import LLM
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM

# Configure logging for error tracking
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


class OllamaHandler(LLM):
    """
    Acts as an agent in LangChain using the Ollama model.

    Attributes:
        model (str): The name of the model to be used.
        llm (Optional[OllamaLLM]): The instantiated Ollama model.
    """
    model: str = Field(default="llama3.2")
    llm: Optional[OllamaLLM] = None

    def __init__(self, **data):
        """
        Initializes the OllamaHandler instance and binds the model.

        Args:
            **data: Arbitrary keyword arguments for model configuration.
        """
        super().__init__(**data)
        object.__setattr__(self, "llm", self.get_llm())

    def get_llm(self) -> Optional[OllamaLLM]:
        """
        Initializes and returns the Ollama model.

        Returns:
            Optional[OllamaLLM]: An instance of the Ollama model, or None if initialization fails.
        """
        try:
            return OllamaLLM(
                model=self.model,
                system_message="Analyze the text and provide a response.",
                return_direct=True,
            )
        except ValueError as e:
            LOGGER.error(f"Error initializing the model: {e}")
            return None

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Invokes the model using LangChain.

        Args:
            prompt (str): The input prompt for the model.
            stop (Optional[List[str]]): A list of stop words.
            **kwargs: Additional arguments for invocation.

        Returns:
            str: The model's response or an error message.
        """
        return self.explain_question_mark(prompt)

    def explain_question_mark(self, question: str) -> str:
        """
        Executes a query and returns the result.

        Args:
            question (str): The input query.

        Returns:
            str: The response from the model or an error message if execution fails.
        """
        if self.llm is None:
            return "Error: Model initialization failed."

        try:
            response = self.llm.invoke(question)
            return response
        except Exception as e:
            LOGGER.error(f"Error executing the model: {e}")
            return f"Error retrieving response: {e}"

    @property
    def _llm_type(self) -> str:
        """
        Returns the type of LLM used.

        Returns:
            str: The string identifier for the LLM type.
        """
        return "custom_ollama"


def main():
    """
    Entry point for executing the script.
    """
    logging.info("LLM Handler script executed successfully!")


if __name__ == "__main__":
    main()
