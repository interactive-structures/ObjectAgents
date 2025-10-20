import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI


class LLMAgent(ABC):
    def __init__(self, prompt_path: str, model: str = "gpt-4o-mini"):
        """
        Initialize the LLM agent with common configuration.

        Args:
            prompt_path: Path to the YAML prompt configuration file
            model: OpenAI model to use
            api_key: OpenAI API key (if None, uses environment variable)
        """
        self.prompt_path = Path(prompt_path)
        self.model = model
        self.prompt_data = self._load_prompt()
        self.system_message_template = self.prompt_data["system_message"]
        self.user_message_template = self.prompt_data["user_message"]

        assert "OPENAI_API_KEY" in os.environ, "no openai api key found"
        self.client = AsyncOpenAI()

    def _load_prompt(self) -> dict[str, Any]:
        """Load the prompt configuration from YAML file."""
        try:
            with open(self.prompt_path, encoding="utf-8") as file:
                return yaml.safe_load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}") from e

    @abstractmethod
    async def query(self, *args: Any, **kwargs: Any) -> Any: ...
