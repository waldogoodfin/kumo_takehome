from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.config import get_settings

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.api_key = self.settings.openai_api_key
        if self.api_key and openai:
            openai.api_key = self.api_key

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and openai)

    def get_client(self):
        if not self.enabled:
            raise RuntimeError("LLM client not enabled")
        return openai.OpenAI(api_key=self.api_key)

    def get_embedding_client(self):
        if not self.enabled:
            raise RuntimeError("LLM client not enabled")
        return openai.Embeddings

    def chat(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            client = self.get_client()
            response = client.chat.completions.create(
                model=self.settings.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception:
            return None

    def parse_json(self, prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        raw = self.chat(prompt, temperature=temperature)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}


_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
