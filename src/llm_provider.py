from __future__ import annotations
import os, json, re
from typing import Any, Dict, Optional
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor:
    - tries direct json.loads
    - else finds first {...} block
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # find JSON object block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in model output: {text[:250]}")
    return json.loads(m.group(0))

def groq_chat_json(system: str, user: str, temperature: float = 0.2, max_tokens: int = 1200) -> Dict[str, Any]:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Set it as an environment variable.")

    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    txt = resp.choices[0].message.content or ""
    return _extract_json(txt)