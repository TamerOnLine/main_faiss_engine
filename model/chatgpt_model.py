import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("YOUR_OPENAI_API_KEY")

# Configure logging for error tracking
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


class ChatGPTHandler(BaseModel):
    """
    LangChain wrapper for ChatGPT.
    """

    model: str = Field(default="gpt-4")  # Can be changed to "gpt-3.5-turbo" for faster & cheaper responses
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")  # Fetch API Key from .env
    temperature: float = Field(default=0.7)  # Controls response variability
    llm: Optional[ChatOpenAI] = None

    def __init__(self, **data):
        """
        Initialize ChatGPT instance.
        """
        super().__init__(**data)

        if not self.api_key:
            LOGGER.error("API Key is missing. Please ensure `OPENAI_API_KEY` is set in the .env file.")
        else:
            self.llm = self.get_llm()

    def get_llm(self) -> Optional[ChatOpenAI]:
        """
        Initialize ChatGPT model.
        """
        if not self.api_key:
            return None

        try:
            return ChatOpenAI(
                model_name=self.model,
                openai_api_key=self.api_key,
                temperature=self.temperature
            )
        except Exception as e:
            LOGGER.error(f"Error initializing ChatGPT: {e}")
            return None

    def ask_question(self, question: str) -> str:
        """
        Send a query to ChatGPT and return the response.
        """
        if self.llm is None:
            return "Error: ChatGPT model initialization failed."

        try:
            response = self.llm.invoke(question)
            return response
        except Exception as e:
            LOGGER.error(f"Error while querying ChatGPT: {e}")
            return f"Error retrieving response: {e}"