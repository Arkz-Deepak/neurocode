"""
Inference interface for local and remote models (Ollama, vLLM, HuggingFace, and Mock).
"""

import json
import urllib.request
from typing import Dict, Any, Optional
from src.agent.patcher import SYSTEM_PROMPT


class BaseLLMClient:
    def generate(self, user_prompt: str, temperature: float = 0.2) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(BaseLLMClient):
    """Connects to any local OpenAI-compatible endpoint (Ollama, vLLM, or llama.cpp server)."""
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "gemma2:9b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, user_prompt: str, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 1024,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                res_json = json.loads(response.read().decode("utf-8"))
                return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error querying local LLM at {url}: {str(e)}"


class MockLLMClient(BaseLLMClient):
    """Simulates an LLM response for local testing and benchmark assertions."""
    def __init__(self, predefined_response: Optional[str] = None):
        self.predefined_response = predefined_response

    def generate(self, user_prompt: str, temperature: float = 0.2) -> str:
        if self.predefined_response:
            return self.predefined_response
        return (
            "FILE: auth/user.py\n"
            "<<<<<<< SEARCH\n"
            "    if len(password) < 8:\n"
            "        return False\n"
            "=======\n"
            "    if not password or len(password) < 8:\n"
            "        return False\n"
            ">>>>>>>"
        )
