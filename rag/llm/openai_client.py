from typing import Optional
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag.pipeline.rag_answer import LLMClient

try:
    from openai import OpenAI
except ImportError as e:
    OpenAI = None


class OpenAILLMClient(LLMClient):
    """
    Simple OpenAI-based implementation of LLMClient.
    Uses chat completion API for generating answers.

    Features:
    - Supports all OpenAI chat models (GPT-4, GPT-3.5, etc.)
    - Configurable temperature
    - Automatic API key loading from environment
    - Error handling for API failures
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
    ):
        """
        Initialize OpenAI LLM client.

        Args:
            model: OpenAI model name (default: gpt-4o-mini)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            temperature: Sampling temperature (0.0-2.0, default: 0.2)

        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not provided
        """
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. "
                "Install with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Either set the environment variable or pass api_key parameter."
            )

        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """
        Generate a response from OpenAI given a full prompt string.

        Args:
            prompt: Complete prompt string with system instructions, context, and query

        Returns:
            Generated answer string from the LLM

        Raises:
            Exception: If OpenAI API call fails
        """
        try:
            # Using the chat.completions API
            # Since PromptBuilder already creates a complete prompt,
            # we feed it as a user message with a simple system message
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                temperature=self.temperature,
            )

            # Extract and return the response content
            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
