from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional

import requests
import time
import random

# Global retry/backoff configuration (no hardcoded literals in logic)
LLM_MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", "3"))
LLM_BACKOFF_BASE_SEC: float = float(os.environ.get("LLM_BACKOFF_BASE_SEC", "1.0"))
LLM_BACKOFF_JITTER_SEC: float = float(os.environ.get("LLM_BACKOFF_JITTER_SEC", "0.25"))


class LLMClient:
    """Minimal LLM client supporting Ollama, OpenAI, and OpenAI-compatible hosts (e.g., LM Studio).

    Providers:
    - "ollama": local Ollama server (default host http://localhost:11434)
    - "openai": OpenAI API (requires OPENAI_API_KEY)
    - "lmstudio" or "openai_compat": OpenAI-compatible APIs (default host http://localhost:1234)

    Env overrides:
    - OPENAI_API_KEY, OLLAMA_HOST, LMSTUDIO_HOST, LMSTUDIO_API_KEY
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        timeout_s: int = 120,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self.provider = provider.lower().strip()
        self.model = model
        # Prefer explicit api_key, then provider-specific envs, then OPENAI_API_KEY
        self.api_key = (
            api_key
            or (os.environ.get("LMSTUDIO_API_KEY") if self.provider in ("lmstudio", "openai_compat") else None)
            or os.environ.get("OPENAI_API_KEY")
        )
        # Provider-specific default hosts
        if host:
            self.host = host
        elif self.provider == "ollama":
            self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        elif self.provider in ("lmstudio", "openai_compat"):
            self.host = os.environ.get("LMSTUDIO_HOST", "http://localhost:1234")
        else:
            self.host = None
        self.timeout_s = timeout_s
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        if self.provider == "ollama":
            return self._complete_ollama(prompt, system)
        elif self.provider == "openai":
            return self._complete_openai(prompt, system)
        elif self.provider in ("lmstudio", "openai_compat"):
            return self._complete_openai_compat(prompt, system)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _complete_ollama(self, prompt: str, system: Optional[str]) -> str:
        url = self.host.rstrip("/") + "/api/chat"
        headers = {"Content-Type": "application/json"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options = {
            "temperature": self.temperature,
            "num_ctx": 8192,
        }
        if self.max_tokens:
            options["num_predict"] = int(self.max_tokens)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        data = self._post_with_retries(url, headers, payload)
        # Ollama chat returns {"message": {"content": "..."}}
        content = data.get("message", {}).get("content", "")
        if not content:
            # Fallback: /api/generate
            url = self.host.rstrip("/") + "/api/generate"
            payload = {
                "model": self.model,
                "prompt": (system + "\n\n" if system else "") + prompt,
                "stream": False,
                "options": options,
            }
            data = self._post_with_retries(url, headers, payload)
            content = data.get("response", "")
        return content

    def _complete_openai(self, prompt: str, system: Optional[str]) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set for OpenAI provider")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(self.max_tokens),
        }
        data = self._post_with_retries(url, headers, payload)
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0]["message"]["content"]

    def _complete_openai_compat(self, prompt: str, system: Optional[str]) -> str:
        if not self.host:
            raise RuntimeError("Host must be set for OpenAI-compatible provider (e.g., LM Studio)")
        # LM Studio defaults to http://localhost:1234 and usually supports /v1/chat/completions
        base = self.host.rstrip("/")
        url = base + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(self.max_tokens),
        }
        data = self._post_with_retries(url, headers, payload)
        choices = data.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content", "")

    def _post_with_retries(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # pragma: no cover
                last_err = e
                if attempt >= LLM_MAX_RETRIES:
                    break
                backoff = (LLM_BACKOFF_BASE_SEC * (2 ** attempt)) + random.random() * LLM_BACKOFF_JITTER_SEC
                time.sleep(backoff)
        # If we get here, retries exhausted
        if last_err:
            raise last_err
        return {}
