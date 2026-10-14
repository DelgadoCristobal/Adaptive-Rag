from typing import Any, Dict, Optional
import os
import re
import json
import shutil
import logging
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

from src.config import settings

load_dotenv()
logger = logging.getLogger(__name__)

PROVIDER_PREFIXES = ("ollama/", "gemini/", "groq/", "openai/")


def qualify_model(model: Optional[str], provider: Optional[str] = None) -> str:
    """Return `model` with a LiteLLM provider prefix, defaulting to the configured provider."""
    model = model or settings.llm_model
    if model.startswith(PROVIDER_PREFIXES):
        return model
    return f"{provider or settings.llm_provider}/{model}"


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse the outermost JSON object in `text` (tolerates Markdown fences and chatter); None if absent/invalid."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class CLIUnavailableError(RuntimeError):
    """The subscription CLI is missing or not logged in; retrying will not help."""


class GeminiCLIClient:
    """Judge backend that shells out to a subscription-authenticated CLI (no API key, no API quota).

    Detects `agy`/`antigravity` (Antigravity CLI) or `gemini` (Gemini CLI) on PATH, the npm global dir, or the
    `JUDGE_CLI` override. The prompt travels via stdin (UTF-8), so quoting and special characters are never an issue.
    """

    CANDIDATES = ("agy", "antigravity", "gemini")
    TIMEOUT_SECONDS = 90
    AUTH_MARKERS = ("ineligibletier", "no longer supported", "not logged in", "login", "authenticat", "unauthorized")

    def __init__(self, binary: Optional[str] = None):
        self.binary = binary or self.detect()

    @classmethod
    def detect(cls) -> Optional[str]:
        override = os.getenv("JUDGE_CLI")
        names = (override,) if override else cls.CANDIDATES
        extra = os.pathsep.join(  # per-user install dirs that the installers do not always add to PATH
            [os.path.join(os.getenv("LOCALAPPDATA", ""), "agy", "bin"), os.path.join(os.getenv("APPDATA", ""), "npm")]
        )
        for name in names:
            found = shutil.which(name) or shutil.which(name, path=extra)
            if found:
                return found
        return None

    @property
    def available(self) -> bool:
        return self.binary is not None

    def complete(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
        if not self.binary:
            raise CLIUnavailableError(
                "No subscription CLI found (agy / antigravity / gemini). Install it and log in, or set JUDGE_CLI."
            )
        model = model or os.getenv("JUDGE_CLI_MODEL")
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        if Path(self.binary).stem.lower() in ("agy", "antigravity"):
            # Agentic CLI: the prompt goes in the argument (stdin makes it try shell tools); read-only plan mode.
            full_prompt = f"Do not use any tools. Answer directly.\n\n{full_prompt}"
            cmd, stdin_text = [self.binary, "-p", full_prompt, "--mode", "plan"], None
        else:
            cmd, stdin_text = [self.binary, "-p", "Answer the request provided on stdin.", "-o", "text"], full_prompt
        if model:
            cmd += ["--model" if stdin_text is None else "-m", model]
        try:
            proc = subprocess.run(
                cmd, input=stdin_text, stdin=None if stdin_text else subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=self.TIMEOUT_SECONDS, cwd=tempfile.gettempdir(),  # neutral cwd: no project context files
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"{Path(self.binary).stem} CLI timed out after {self.TIMEOUT_SECONDS}s") from e
        out, err = (proc.stdout or "").strip(), (proc.stderr or "").strip()
        if proc.returncode != 0 or not out:
            detail = (err or out)[:300]
            if any(marker in detail.lower() for marker in self.AUTH_MARKERS):
                raise CLIUnavailableError(f"{Path(self.binary).stem} CLI is not usable (login/migration needed): {detail}")
            raise RuntimeError(f"{Path(self.binary).stem} CLI failed (exit {proc.returncode}): {detail}")
        return out


class LLMClient:
    """Unified, provider-agnostic LLM interface supporting Gemini, Ollama, Groq, and OpenAI."""

    def __init__(self, default_model: Optional[str] = None):
        self.default_model = qualify_model(default_model or os.getenv("LLM_MODEL"))
        self.ollama_api_base = os.getenv("OLLAMA_API_BASE", settings.ollama_api_base).rstrip("/")
        self.num_gpu = int(os.getenv("OLLAMA_NUM_GPU", settings.ollama_num_gpu))
        self.num_thread = int(os.getenv("OLLAMA_NUM_THREAD", settings.ollama_num_thread))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", settings.ollama_num_ctx))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", settings.ollama_num_predict))
        self._litellm = None

    def _get_litellm(self):
        if self._litellm is None:
            import litellm
            self._litellm = litellm
        return self._litellm

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Sends a completion request via LiteLLM, with a direct HTTP fallback for local Ollama.

        Args:
            prompt: User message prompt.
            system_prompt: Optional system instruction.
            model: Target model string (e.g. 'ollama/gemma:2b'); defaults to the client's model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate (limits KV cache and generation runaway).
        """
        target_model = qualify_model(model or self.default_model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Hardware & runtime options for local Ollama execution
        ollama_options = {
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "num_ctx": self.num_ctx,
            "num_predict": max_tokens or self.num_predict,
            "temperature": temperature,
        }

        try:
            kwargs: Dict[str, Any] = {"model": target_model, "messages": messages}

            if target_model.startswith("ollama/"):
                kwargs["api_base"] = self.ollama_api_base
                kwargs["extra_body"] = {"options": ollama_options}
            if not target_model.startswith("gemini/"):
                kwargs["temperature"] = temperature
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens

            response = self._get_litellm().completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            if target_model.startswith("ollama/"):
                try:
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    body = json.dumps({
                        "model": target_model.removeprefix("ollama/"),
                        "prompt": full_prompt,
                        "stream": False,
                        "options": ollama_options,
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        f"{self.ollama_api_base}/api/generate", data=body, headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()
                except Exception as fallback_e:
                    logger.error("Direct Ollama fallback error for model %s: %s", target_model, fallback_e)

            logger.error("LLM completion error using model %s: %s", target_model, e)
            raise
