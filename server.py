from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
import asyncio

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Change to INFO level to show more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn

# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator",
        ]

        if hasattr(record, "msg") and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True


# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())


# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)


# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(
            ColorizedFormatter("%(asctime)s - %(levelname)s - %(message)s")
        )

# Also persist WARNING+ logs (errors, retries, 422s) to a file so failures can be
# inspected after the fact — even when the server runs quietly (log_level="error")
# or in the background. Path is overridable via PROXY_LOG_FILE.
try:
    _proxy_log_path = os.environ.get("PROXY_LOG_FILE", "proxy_errors.log")
    _file_handler = logging.FileHandler(_proxy_log_path)
    _file_handler.setLevel(logging.WARNING)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    _file_handler.addFilter(MessageFilter())
    logging.getLogger().addHandler(_file_handler)
    logger.warning(f"Proxy diagnostic log enabled at: {_proxy_log_path}")
except Exception as _log_setup_err:  # pragma: no cover
    logger.warning(f"Could not set up file logging: {_log_setup_err}")

app = FastAPI()


# Log (and Anthropic-format) request validation failures. FastAPI returns 422
# *before* the endpoint runs, so these never appear in the normal request logs —
# which makes silent 422s look like the model being "temporarily unavailable" in
# Claude Code. This surfaces exactly which field/block was rejected.
from fastapi.exceptions import RequestValidationError  # noqa: E402


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        errors = exc.errors()
        # Trim verbose 'input' payloads so the log stays readable.
        brief = []
        for err in errors:
            brief.append(
                {
                    "loc": err.get("loc"),
                    "type": err.get("type"),
                    "msg": err.get("msg"),
                }
            )
        logger.error(
            f"⛔ 422 validation error on {request.method} {request.url.path}: "
            f"{json.dumps(brief)}"
        )
    except Exception as log_err:
        logger.error(f"422 validation error (failed to format details): {log_err}")
    return JSONResponse(
        status_code=422,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Request validation failed; see proxy logs for details.",
            },
        },
    )

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get Vertex AI project and location from environment (if set)
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")

# Option to use Gemini API key instead of ADC for Vertex AI
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# Get OpenAI base URL from environment (if set)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
# Tier mapping (Anthropic model family -> backend model):
#   opus   -> BIG_MODEL    (highest tier, e.g. fugu-ultra)
#   sonnet -> SMALL_MODEL  (mid/daily-driver tier, e.g. fugu)
#   haiku  -> HAIKU_MODEL  (low tier; defaults to a free Gemini model)
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")
# Low-tier model used for Haiku. Defaults to Google's free Gemini Flash, which
# is the closest free analog to Claude Haiku. Requires GEMINI_API_KEY when a
# gemini-* model is used.
HAIKU_MODEL = os.environ.get("HAIKU_MODEL", "gemini-2.5-flash")

# --- Exa AI web search configuration ---------------------------------------
# The backend (e.g. api.sakana.ai) cannot run Anthropic's server-side web
# search, so this proxy executes the search itself using Exa AI and feeds the
# results back to the model. Set EXA_API_KEY to enable it.
EXA_API_KEY = os.environ.get("EXA_API_KEY")
EXA_SEARCH_URL = os.environ.get("EXA_SEARCH_URL", "https://api.exa.ai/search")
EXA_NUM_RESULTS = int(os.environ.get("EXA_NUM_RESULTS", "5"))
EXA_TEXT_MAX_CHARS = int(os.environ.get("EXA_TEXT_MAX_CHARS", "2000"))
# Safety cap on how many web-search round-trips a single request may trigger.
WEB_SEARCH_MAX_USES = int(os.environ.get("WEB_SEARCH_MAX_USES", "5"))

# Canonical function name the proxy exposes to the backend for web search, plus
# the set of names/types that identify an incoming Anthropic web-search tool.
WEB_SEARCH_FUNCTION_NAME = "web_search"
WEB_SEARCH_TOOL_NAMES = {"web_search", "websearch"}

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini",  # Added default small model
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def _provider_prefix_for(model_name: str) -> str:
    """Attach the correct LiteLLM provider prefix for a bare backend model name."""
    if model_name.startswith(("openai/", "gemini/", "anthropic/")):
        return model_name
    if model_name in GEMINI_MODELS:
        return f"gemini/{model_name}"
    # Everything else (incl. Sakana fugu/fugu-ultra) goes through the OpenAI path.
    return f"openai/{model_name}"


def response_model_name(original_request) -> str:
    """The model name to report back to the client.

    Prefer the original Claude name the client sent (e.g. "claude-sonnet-4-6")
    over the mapped backend name (e.g. "openai/fugu"), so the mapped name does
    not leak back into the client's later requests (count_tokens, etc.).
    """
    original = getattr(original_request, "original_model", None)
    if original and original != "unknown":
        return original
    return original_request.model


def map_anthropic_model(v: str):
    """Map an incoming Anthropic model name to a backend model + provider prefix.

    Tiering:
      haiku  -> HAIKU_MODEL  (low tier, free)
      sonnet -> SMALL_MODEL  (mid tier)
      opus   -> BIG_MODEL    (high tier)
    Returns (new_model, mapped: bool).
    """
    # Strip any existing provider prefix for matching.
    clean_v = v
    if clean_v.startswith("anthropic/"):
        clean_v = clean_v[10:]
    elif clean_v.startswith("openai/"):
        clean_v = clean_v[7:]
    elif clean_v.startswith("gemini/"):
        clean_v = clean_v[7:]

    # "Just an Anthropic proxy" mode: keep the model, only add the prefix.
    if PREFERRED_PROVIDER == "anthropic":
        return f"anthropic/{clean_v}", True

    low = clean_v.lower()
    if "haiku" in low:
        return _provider_prefix_for(HAIKU_MODEL), True
    if "sonnet" in low:
        return _provider_prefix_for(SMALL_MODEL), True
    if "opus" in low:
        return _provider_prefix_for(BIG_MODEL), True

    # Not a tiered Claude name: add a prefix if it's a known backend model.
    if clean_v in GEMINI_MODELS and not v.startswith("gemini/"):
        return f"gemini/{clean_v}", True
    if clean_v in OPENAI_MODELS and not v.startswith("openai/"):
        return f"openai/{clean_v}", True

    return v, False


# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(
                    f"Removing unsupported format '{schema['format']}' for string type in Gemini schema."
                )
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(
            schema.items()
        ):  # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema


# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class ContentBlockServerToolUse(BaseModel):
    type: Literal["server_tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockWebSearchToolResult(BaseModel):
    type: Literal["web_search_tool_result"]
    tool_use_id: str
    content: Union[List[Dict[str, Any]], Dict[str, Any]]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
                ContentBlockServerToolUse,
                ContentBlockWebSearchToolResult,
            ]
        ],
    ]

    @field_validator("content", mode="before")
    def _normalize_content_blocks(cls, v):
        """Tolerate content block types the proxy doesn't explicitly model.

        Claude Code replays full history on every request — including blocks the
        proxy has no schema for (e.g. ``thinking`` / ``redacted_thinking`` from
        extended thinking, ``document``, etc.). Without this, those blocks make
        Pydantic reject the whole request with 422, which Claude Code surfaces as
        the model being "temporarily unavailable" (notably for the auto-mode
        safety classifier). We drop reasoning blocks and coerce other unknown
        blocks to text so validation always succeeds.
        """
        if not isinstance(v, list):
            return v
        known = {
            "text",
            "image",
            "tool_use",
            "tool_result",
            "server_tool_use",
            "web_search_tool_result",
        }
        cleaned = []
        for item in v:
            if not isinstance(item, dict):
                # Already a validated block model (e.g. from count_tokens reuse).
                cleaned.append(item)
                continue
            t = item.get("type")
            if t in known:
                cleaned.append(item)
            elif t in ("thinking", "redacted_thinking"):
                # Reasoning blocks are not forwarded to non-Anthropic backends.
                continue
            else:
                # Unknown block: keep any text so context isn't lost; else drop.
                text = item.get("text")
                if not isinstance(text, str):
                    inner = item.get("content")
                    text = inner if isinstance(inner, str) else None
                if isinstance(text, str) and text:
                    cleaned.append({"type": "text", "text": text})
        return cleaned


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    # input_schema is required for normal (client-side) function tools, but
    # Anthropic server tools such as web search are sent WITHOUT it, e.g.
    #   {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    # so we make it optional and tolerate the extra server-tool fields.
    input_schema: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    max_uses: Optional[int] = None
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    user_location: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}

    def is_web_search(self) -> bool:
        return is_web_search_tool_dict(
            self.dict() if hasattr(self, "dict") else dict(self)
        )


class ThinkingConfig(BaseModel):
    # Anthropic wire format: {"type": "enabled", "budget_tokens": N}
    type: Optional[str] = "enabled"   # "enabled" | "disabled"
    budget_tokens: Optional[int] = None
    # Legacy field — kept for backward compat
    enabled: Optional[bool] = None

    def is_enabled(self) -> bool:
        if self.type is not None:
            return self.type == "enabled"
        return self.enabled is True


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator("model")
    def validate_model_field(cls, v, info):  # Renamed to avoid conflict
        original_model = v

        logger.debug(
            f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', "
            f"OPUS/BIG='{BIG_MODEL}', SONNET/SMALL='{SMALL_MODEL}', HAIKU='{HAIKU_MODEL}'"
        )

        new_model, mapped = map_anthropic_model(v)

        if mapped:
            logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
            # If no mapping occurred and no prefix exists, log warning or decide default
            if not v.startswith(("openai/", "gemini/", "anthropic/")):
                logger.warning(
                    f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is."
                )
            new_model = v  # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values["original_model"] = original_model

        return new_model


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator("model")
    def validate_model_token_count(cls, v, info):  # Renamed to avoid conflict
        # Shares the same tiering logic as MessagesRequest.
        original_model = v

        logger.debug(
            f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', "
            f"OPUS/BIG='{BIG_MODEL}', SONNET/SMALL='{SMALL_MODEL}', HAIKU='{HAIKU_MODEL}'"
        )

        new_model, mapped = map_anthropic_model(v)

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
            if not v.startswith(("openai/", "gemini/", "anthropic/")):
                logger.warning(
                    f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is."
                )
            new_model = v  # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values["original_model"] = original_model

        return new_model


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[
        Union[
            ContentBlockText,
            ContentBlockToolUse,
            ContentBlockServerToolUse,
            ContentBlockWebSearchToolResult,
        ]
    ]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path

    # Process the request and get the response
    response = await call_next(request)

    # Log EVERY request (method, path, status) to the diagnostic file so we can
    # see endpoints the proxy doesn't implement (404s) or unexpected statuses —
    # these never reach the per-endpoint handlers and are otherwise invisible.
    # Logged at WARNING so it always reaches proxy_errors.log during diagnosis.
    try:
        status = getattr(response, "status_code", "?")
        logger.warning(f"REQ {method} {path} -> {status}")
    except Exception:
        pass

    return response


# Not using validation function as we're using the environment API key


# Transient upstream conditions worth retrying. These commonly cause Claude
# Code to report a model as "temporarily unavailable" (e.g. the auto-mode
# safety classifier failing before an Edit).
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
RETRYABLE_EXCEPTION_NAMES = {
    "RateLimitError",
    "Timeout",
    "APITimeoutError",
    "APIConnectionError",
    "ServiceUnavailableError",
    "InternalServerError",
    "OverloadedError",
}


def _is_retryable_error(e: Exception) -> bool:
    """Decide whether an upstream exception is a transient error worth retrying."""
    status = getattr(e, "status_code", None)
    if status in RETRYABLE_STATUS_CODES:
        return True
    if type(e).__name__ in RETRYABLE_EXCEPTION_NAMES:
        return True
    # Some connection/timeout errors carry no status_code and a generic type;
    # fall back to a conservative message sniff.
    msg = str(e).lower()
    transient_markers = (
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection error",
        "temporarily unavailable",
        "overloaded",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    return any(marker in msg for marker in transient_markers)


async def retry_with_backoff(coro_func, max_retries: int = 4, base_delay: float = 1.0):
    """Retry an async coroutine with exponential backoff + jitter on transient errors.

    Covers HTTP 408/409/425/429/500/502/503/504/529 plus rate-limit, timeout and
    connection errors — the failure modes that otherwise surface to the client as
    "temporarily unavailable".
    """
    import random

    for attempt in range(max_retries):
        try:
            return await coro_func()
        except Exception as e:
            if _is_retryable_error(e) and attempt < max_retries - 1:
                status = getattr(e, "status_code", None) or type(e).__name__
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    f"Transient upstream error [{status}] "
                    f"(attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                raise


def parse_text_tool_calls(text: str):
    """Parse tool calls embedded as <tool_call>...</tool_call> tags in text content.

    Models such as Qwen and DeepSeek emit tool calls inline in the text rather
    than via the native ``tool_calls`` field.  Returns a tuple of
    (cleaned_text, list_of_tool_use_blocks).
    """
    if not text or "<tool_call>" not in text:
        return text, []

    tool_uses = []
    pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def _replace(match):
        try:
            call_data = json.loads(match.group(1))
            name = call_data.get("name", "")
            arguments = call_data.get("arguments", call_data.get("parameters", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            tool_uses.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": name,
                    "input": arguments,
                }
            )
            return ""
        except (json.JSONDecodeError, KeyError, AttributeError):
            return match.group(0)  # Leave unparseable tags in the text

    cleaned = pattern.sub(_replace, text).strip()
    return cleaned, tool_uses


def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"


# ---------------------------------------------------------------------------
# Web search (Exa AI) support
# ---------------------------------------------------------------------------
def is_web_search_tool_dict(tool_dict: Dict[str, Any]) -> bool:
    """Return True if an Anthropic tool definition represents web search.

    Matches both the native server-tool shape
    ({"type": "web_search_20250305", "name": "web_search"}) and a plain
    function tool that happens to be named web_search / WebSearch.
    """
    if not isinstance(tool_dict, dict):
        return False
    t = tool_dict.get("type")
    if isinstance(t, str) and t.startswith("web_search"):
        return True
    name = tool_dict.get("name")
    if isinstance(name, str) and name.lower() in WEB_SEARCH_TOOL_NAMES:
        return True
    return False


# JSON schema advertised to the backend for the web_search function tool.
WEB_SEARCH_PARAMETERS_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to look up on the web.",
        }
    },
    "required": ["query"],
}


def exa_search(
    query: str,
    num_results: int = EXA_NUM_RESULTS,
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run a web search via the Exa AI API.

    Returns a dict: {"results": [ {title, url, published_date, snippet} ], "error": Optional[str]}.
    """
    if not EXA_API_KEY:
        return {
            "results": [],
            "error": "Web search is not configured: EXA_API_KEY is not set on the proxy.",
        }

    payload: Dict[str, Any] = {
        "query": query,
        "type": "auto",
        "numResults": max(1, int(num_results or EXA_NUM_RESULTS)),
        "contents": {
            "text": {"maxCharacters": EXA_TEXT_MAX_CHARS},
            "highlights": {"numSentences": 3, "highlightsPerUrl": 2},
        },
    }
    if allowed_domains:
        payload["includeDomains"] = allowed_domains
    if blocked_domains:
        payload["excludeDomains"] = blocked_domains

    headers = {
        "x-api-key": EXA_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(EXA_SEARCH_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        logger.error(f"Exa search HTTP error {e.response.status_code}: {body}")
        return {"results": [], "error": f"Exa search failed: HTTP {e.response.status_code}"}
    except Exception as e:
        logger.error(f"Exa search error: {e}")
        return {"results": [], "error": f"Exa search failed: {e}"}

    results = []
    for item in data.get("results", []) or []:
        snippet = ""
        highlights = item.get("highlights") or []
        if highlights:
            snippet = " … ".join(h for h in highlights if h)
        if not snippet:
            text = item.get("text") or ""
            snippet = text[:EXA_TEXT_MAX_CHARS]
        results.append(
            {
                "title": item.get("title") or item.get("url") or "Untitled",
                "url": item.get("url", ""),
                "published_date": item.get("publishedDate") or item.get("published_date"),
                "author": item.get("author"),
                "snippet": snippet.strip(),
            }
        )
    return {"results": results, "error": None}


def format_search_results_for_model(query: str, search: Dict[str, Any]) -> str:
    """Build a compact, model-friendly string from Exa results."""
    if search.get("error"):
        return f"Web search for '{query}' failed: {search['error']}"
    results = search.get("results", [])
    if not results:
        return f"Web search for '{query}' returned no results."

    lines = [f"Search results for: {query}", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title')}")
        lines.append(f"URL: {r.get('url')}")
        if r.get("published_date"):
            lines.append(f"Published: {r.get('published_date')}")
        if r.get("snippet"):
            lines.append(f"Snippet: {r.get('snippet')}")
        lines.append("")
    return "\n".join(lines).strip()


def build_web_search_result_block(tool_id: str, search: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single Anthropic web_search_tool_result block from an Exa result."""
    import base64

    if search.get("error"):
        return {
            "type": "web_search_tool_result",
            "tool_use_id": tool_id,
            "content": {
                "type": "web_search_tool_result_error",
                "error_code": "unavailable",
            },
        }
    result_items = []
    for r in search.get("results", []):
        snippet = r.get("snippet") or ""
        encrypted = base64.b64encode(snippet.encode("utf-8")).decode("ascii")
        result_items.append(
            {
                "type": "web_search_result",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "encrypted_content": encrypted,
                "page_age": r.get("published_date"),
            }
        )
    return {
        "type": "web_search_tool_result",
        "tool_use_id": tool_id,
        "content": result_items,
    }


def build_web_search_result_blocks(search_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert recorded searches into Anthropic server_tool_use + web_search_tool_result blocks."""
    blocks: List[Dict[str, Any]] = []
    for rec in search_records:
        query = rec.get("query", "")
        tool_id = rec.get("tool_use_id") or f"srvtoolu_{uuid.uuid4().hex[:24]}"
        blocks.append(
            {
                "type": "server_tool_use",
                "id": tool_id,
                "name": "web_search",
                "input": {"query": query},
            }
        )
        blocks.append(build_web_search_result_block(tool_id, rec.get("search", {})))
    return blocks


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format

    messages = []

    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, "type") and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"

            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})

    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool,
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(
                block.type == "tool_result"
                for block in content
                if hasattr(block, "type")
            ):
                # Convert each tool_result block to a proper OpenAI/LiteLLM
                # {"role": "tool", "tool_call_id": ..., "content": ...} message.
                # LiteLLM then translates these correctly to whichever backend is in
                # use (Anthropic tool_result blocks, OpenAI tool role, etc.).
                extra_text = ""
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            extra_text += block.text + "\n"
                        elif block.type == "tool_result":
                            tool_id = (
                                block.tool_use_id
                                if hasattr(block, "tool_use_id")
                                else ""
                            )
                            result_content = parse_tool_result_content(
                                block.content if hasattr(block, "content") else None
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "content": result_content or "",
                                }
                            )
                # Any accompanying user text goes in a separate user message
                if extra_text.strip():
                    messages.append({"role": "user", "content": extra_text.strip()})
            else:
                # Regular handling for other message types.
                # For assistant messages: tool_use blocks are emitted as OpenAI-style
                # tool_calls so that LiteLLM can translate them for any backend.
                processed_content = []
                openai_tool_calls = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append(
                                {"type": "text", "text": block.text}
                            )
                        elif block.type == "image":
                            processed_content.append(
                                {"type": "image", "source": block.source}
                            )
                        elif block.type == "tool_use":
                            # Convert to OpenAI function-call format
                            arguments = block.input
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments)
                            openai_tool_calls.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": arguments,
                                    },
                                }
                            )
                        elif block.type == "tool_result":
                            # Handle different formats of tool result content
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id
                                if hasattr(block, "tool_use_id")
                                else "",
                            }

                            # Process the content field properly
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    # If it's a simple string, create a text block for it
                                    processed_content_block["content"] = [
                                        {"type": "text", "text": block.content}
                                    ]
                                elif isinstance(block.content, list):
                                    # If it's already a list of blocks, keep it
                                    processed_content_block["content"] = block.content
                                else:
                                    # Default fallback
                                    processed_content_block["content"] = [
                                        {"type": "text", "text": str(block.content)}
                                    ]
                            else:
                                # Default empty content
                                processed_content_block["content"] = [
                                    {"type": "text", "text": ""}
                                ]

                            processed_content.append(processed_content_block)

                        elif block.type == "server_tool_use":
                            # Web search request the proxy executed on a prior turn.
                            # Echoed back by the client as history; flatten to text.
                            inp = getattr(block, "input", {}) or {}
                            query = (
                                inp.get("query", "") if isinstance(inp, dict) else ""
                            )
                            processed_content.append(
                                {
                                    "type": "text",
                                    "text": f"[Web search performed: {query}]",
                                }
                            )

                        elif block.type == "web_search_tool_result":
                            # Results of a prior proxy web search, echoed back as
                            # history. Flatten to a short text summary so the model
                            # retains the context without the opaque encrypted blob.
                            rc = getattr(block, "content", None)
                            lines = ["[Web search results]"]
                            if isinstance(rc, list):
                                for item in rc:
                                    if (
                                        isinstance(item, dict)
                                        and item.get("type") == "web_search_result"
                                    ):
                                        title = item.get("title", "")
                                        url = item.get("url", "")
                                        lines.append(f"- {title} ({url})")
                            processed_content.append(
                                {"type": "text", "text": "\n".join(lines)}
                            )

                if openai_tool_calls and msg.role == "assistant":
                    # Emit an assistant message with tool_calls in OpenAI format.
                    text_parts = [
                        b["text"]
                        for b in processed_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "\n".join(text_parts) if text_parts else None,
                            "tool_calls": openai_tool_calls,
                        }
                    )
                else:
                    messages.append({"role": msg.role, "content": processed_content})

    # Cap max_tokens for OpenAI models to their limit of 16384
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith(
        "openai/"
    ) or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(
            f"Capping max_tokens to 16384 for OpenAI/Gemini model (original value: {anthropic_request.max_tokens})"
        )

    # OpenAI-compatible endpoints (e.g. Sakana AI) reject max_completion_tokens < 16.
    # Claude Code sends tiny probe requests (max_tokens=1) for Haiku, so enforce a floor.
    if max_tokens < 16:
        logger.debug(
            f"Raising max_tokens to minimum of 16 (original value: {max_tokens})"
        )
        max_tokens = 16

    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # it understands "anthropic/claude-x" format
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Map thinking config to the correct provider-specific parameter
    if anthropic_request.thinking and anthropic_request.thinking.is_enabled():
        if anthropic_request.model.startswith("anthropic/"):
            # Pass native Anthropic extended thinking params as-is
            litellm_request["thinking"] = anthropic_request.thinking
        else:
            # Straight 1-to-1 mapping: Claude Code effort → reasoning_effort string
            #
            #   Claude Code │ budget_tokens │ reasoning_effort
            #   ────────────────────────────────────────────────
            #   xhigh       │   ≥ 16 000    │  "xhigh"
            #   high        │   ≥  8 000    │  "high"
            #   normal      │   ≥  1 024    │  "medium"
            #   low         │   <  1 024    │  "low"
            #
            # Use extra_body instead of a top-level param so LiteLLM's model
            # validation is bypassed — custom/unknown models (fugu, qwen, etc.)
            # won't raise UnsupportedParamsError.
            budget = anthropic_request.thinking.budget_tokens
            if budget is not None:
                if budget >= 16000:
                    effort = "xhigh"
                elif budget >= 8000:
                    effort = "high"
                elif budget >= 1024:
                    effort = "medium"
                else:
                    effort = "low"
            else:
                effort = "high"  # default when thinking is enabled
            litellm_request.setdefault("extra_body", {})["reasoning_effort"] = effort

    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences

    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p

    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k

    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")
        seen_web_search = False

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, "dict"):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                    logger.error(f"Could not convert tool to dict: {tool}")
                    continue  # Skip this tool if conversion fails

            # Web search is a server tool executed by THIS proxy (via Exa), not
            # by the backend. Expose it to the backend as a normal function tool
            # with a canonical name so we can intercept the call and run Exa.
            if is_web_search_tool_dict(tool_dict):
                if seen_web_search:
                    continue  # avoid duplicate web_search tool definitions
                seen_web_search = True
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": WEB_SEARCH_FUNCTION_NAME,
                            "description": (
                                "Search the public web for up-to-date information. "
                                "Returns a list of relevant results with titles, URLs "
                                "and snippets. Use this whenever the user asks about "
                                "current events or facts you are unsure about."
                            ),
                            "parameters": WEB_SEARCH_PARAMETERS_SCHEMA,
                        },
                    }
                )
                continue

            # Clean the schema if targeting a Gemini model
            input_schema = tool_dict.get("input_schema", {})
            if input_schema is None:
                input_schema = {}
            if is_gemini_model:
                logger.debug(
                    f"Cleaning schema for Gemini tool: {tool_dict.get('name')}"
                )
                input_schema = clean_gemini_schema(input_schema)

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema,  # Use potentially cleaned schema
                },
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools

    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, "dict"):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice

        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]},
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"

    return litellm_request


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any],
    original_request: MessagesRequest,
    search_records: Optional[List[Dict[str, Any]]] = None,
) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""

    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/") :]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/") :]

        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")

        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, "choices") and hasattr(litellm_response, "usage"):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = (
                message.content if message and hasattr(message, "content") else ""
            )
            tool_calls = (
                message.tool_calls
                if message and hasattr(message, "tool_calls")
                else None
            )
            finish_reason = (
                choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            )
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, "id", f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = (
                    litellm_response
                    if isinstance(litellm_response, dict)
                    else litellm_response.dict()
                )
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__
                try:
                    response_dict = (
                        litellm_response.model_dump()
                        if hasattr(litellm_response, "model_dump")
                        else litellm_response.__dict__
                    )
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, "id", f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, "choices", [{}]),
                        "usage": getattr(litellm_response, "usage", {}),
                    }

            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = (
                choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            )
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = (
                choices[0].get("finish_reason", "stop")
                if choices and len(choices) > 0
                else "stop"
            )
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        # Create content list for Anthropic format
        content = []

        # Prepend web search server_tool_use + web_search_tool_result blocks so
        # Claude Code renders the searches the proxy executed on its behalf.
        if search_records:
            content.extend(build_web_search_result_blocks(search_records))

        # Fallback: parse tool calls embedded as <tool_call> tags in text content
        # (used by models like Qwen/DeepSeek that don't emit native tool_calls)
        if content_text and not tool_calls:
            content_text, embedded_tool_uses = parse_text_tool_calls(content_text)
            if embedded_tool_uses:
                logger.debug(f"Parsed {len(embedded_tool_uses)} tool call(s) from text content")
                tool_calls = embedded_tool_uses  # treat them like native tool_calls below

        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})

        # Add tool calls if present (tool_use in Anthropic format)
        # Works for ALL models regardless of provider prefix.
        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")

            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")

                # Already a fully-formed tool_use block (e.g. from parse_text_tool_calls)
                if isinstance(tool_call, dict) and tool_call.get("type") == "tool_use":
                    content.append(tool_call)
                    continue

                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = (
                        getattr(function, "arguments", "{}") if function else "{}"
                    )

                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse tool arguments as JSON: {arguments}"
                        )
                        arguments = {"raw": arguments}

                logger.debug(
                    f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}"
                )

                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": arguments,
                    }
                )

        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default

        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})

        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=response_model_name(original_request),
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

        return anthropic_response

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = (
            f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        )
        logger.error(error_message)

        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=response_model_name(original_request),
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": f"Error converting response: {str(e)}. Please check server logs.",
                }
            ],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
        )


# ---------------------------------------------------------------------------
# In-proxy web search agentic loop
# ---------------------------------------------------------------------------
def _tool_call_fields(tc) -> Dict[str, Any]:
    """Normalize an OpenAI tool_call (dict or object) into {id, name, arguments}."""
    if isinstance(tc, dict):
        fn = tc.get("function", {}) or {}
        return {
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name", "") if isinstance(fn, dict) else "",
            "arguments": (fn.get("arguments", "") if isinstance(fn, dict) else "") or "",
        }
    fn = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", None) or f"toolu_{uuid.uuid4().hex[:24]}",
        "name": getattr(fn, "name", "") if fn else "",
        "arguments": (getattr(fn, "arguments", "") if fn else "") or "",
    }


def _parse_tool_arguments(raw) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"query": str(raw)}


def run_web_search_loop_sync(
    litellm_request: Dict[str, Any],
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
):
    """Drive the backend through web-search tool calls, executing each via Exa.

    Runs synchronously (intended to be called inside a thread executor). Returns
    a tuple of (final_litellm_response, search_records).
    """
    req = dict(litellm_request)
    req["stream"] = False
    req["num_retries"] = 2  # let LiteLLM retry transient backend errors per call
    messages = list(req.get("messages", []))
    req["messages"] = messages

    search_records: List[Dict[str, Any]] = []
    uses = 0
    response = None

    for _ in range(WEB_SEARCH_MAX_USES + 1):
        response = litellm.completion(**req)
        choices = getattr(response, "choices", None) or []
        if not choices:
            break
        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        web_calls = []
        for tc in tool_calls:
            fields = _tool_call_fields(tc)
            if fields["name"] == WEB_SEARCH_FUNCTION_NAME:
                web_calls.append(fields)

        # No web search requested (or budget exhausted) -> this is the final answer.
        if not web_calls or uses >= WEB_SEARCH_MAX_USES:
            break

        # Append an assistant turn containing ONLY the web_search tool calls so
        # every tool_call_id we answer has a matching tool message (OpenAI rule).
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {
                        "name": WEB_SEARCH_FUNCTION_NAME,
                        "arguments": c["arguments"]
                        if isinstance(c["arguments"], str)
                        else json.dumps(c["arguments"]),
                    },
                }
                for c in web_calls
            ],
        }
        messages.append(assistant_msg)

        for c in web_calls:
            args = _parse_tool_arguments(c["arguments"])
            query = args.get("query") or args.get("q") or ""
            logger.debug(f"🔎 Executing Exa web search: {query!r}")
            search = exa_search(
                query,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
            )
            uses += 1
            search_records.append(
                {"tool_use_id": c["id"], "query": query, "search": search}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "name": WEB_SEARCH_FUNCTION_NAME,
                    "content": format_search_results_for_model(query, search),
                }
            )

    return response, search_records


async def synthesize_streaming_response(
    anthropic_response: MessagesResponse, original_request: MessagesRequest
):
    """Emit Anthropic SSE events from an already-computed MessagesResponse.

    Used for the web-search path, where the final answer is produced via a
    (non-streaming) agentic loop but the client requested a stream.
    """
    message_id = anthropic_response.id or f"msg_{uuid.uuid4().hex[:24]}"

    message_data = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": response_model_name(original_request),
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": anthropic_response.usage.input_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    # Normalize content blocks to plain dicts
    blocks = []
    for block in anthropic_response.content:
        if hasattr(block, "dict"):
            blocks.append(block.dict())
        elif isinstance(block, dict):
            blocks.append(block)

    for index, block in enumerate(blocks):
        btype = block.get("type")
        if btype == "text":
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            text = block.get("text", "") or ""
            if text:
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"
        elif btype in ("tool_use", "server_tool_use"):
            start_block = {
                "type": btype,
                "id": block.get("id"),
                "name": block.get("name"),
                "input": {},
            }
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': start_block})}\n\n"
            partial = json.dumps(block.get("input", {}))
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': partial}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"
        else:
            # web_search_tool_result and any other complete block: send whole.
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': block})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"

    usage = {"output_tokens": anthropic_response.usage.output_tokens}
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': anthropic_response.stop_reason or 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    yield "data: [DONE]\n\n"


async def handle_streaming_web_search(
    litellm_request: Dict[str, Any],
    original_request: MessagesRequest,
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
):
    """Run the web-search agentic loop *inside* the SSE stream.

    This emits each ``server_tool_use`` block the moment a search begins (so
    Claude Code shows the live ``WebSearch("…")`` status), then the matching
    ``web_search_tool_result`` block, then finally streams the answer text.
    """
    response_started = False
    try:
        loop = asyncio.get_event_loop()
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        message_data = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": response_model_name(original_request),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        response_started = True

        req = dict(litellm_request)
        req["stream"] = False
        req["num_retries"] = 2  # let LiteLLM retry transient backend errors per call
        messages = list(req.get("messages", []))
        req["messages"] = messages

        idx = 0
        uses = 0
        output_tokens = 0
        stop_reason = "end_turn"

        for _ in range(WEB_SEARCH_MAX_USES + 1):
            response = await loop.run_in_executor(
                None, lambda: litellm.completion(**req)
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                break
            message = choices[0].message
            usage = getattr(response, "usage", None)
            if usage is not None and getattr(usage, "completion_tokens", None):
                output_tokens += usage.completion_tokens

            tool_calls = getattr(message, "tool_calls", None) or []
            web_calls, other_calls = [], []
            for tc in tool_calls:
                f = _tool_call_fields(tc)
                (web_calls if f["name"] == WEB_SEARCH_FUNCTION_NAME else other_calls).append(f)

            # ---- A web search round: emit live blocks, run Exa, loop again ----
            if web_calls and uses < WEB_SEARCH_MAX_USES:
                messages.append(
                    {
                        "role": "assistant",
                        "content": getattr(message, "content", None) or "",
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {
                                    "name": WEB_SEARCH_FUNCTION_NAME,
                                    "arguments": c["arguments"]
                                    if isinstance(c["arguments"], str)
                                    else json.dumps(c["arguments"]),
                                },
                            }
                            for c in web_calls
                        ],
                    }
                )

                for c in web_calls:
                    args = _parse_tool_arguments(c["arguments"])
                    query = args.get("query") or args.get("q") or ""

                    # 1) server_tool_use block -> renders WebSearch("query") live
                    start_block = {
                        "type": "server_tool_use",
                        "id": c["id"],
                        "name": "web_search",
                        "input": {},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': start_block})}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps({'query': query})}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
                    idx += 1

                    # 2) run the actual search
                    logger.debug(f"🔎 [stream] Exa web search: {query!r}")
                    search = await loop.run_in_executor(
                        None,
                        lambda q=query: exa_search(
                            q,
                            allowed_domains=allowed_domains,
                            blocked_domains=blocked_domains,
                        ),
                    )
                    uses += 1

                    # 3) web_search_tool_result block -> renders the results
                    result_block = build_web_search_result_block(c["id"], search)
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': result_block})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
                    idx += 1

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": c["id"],
                            "name": WEB_SEARCH_FUNCTION_NAME,
                            "content": format_search_results_for_model(query, search),
                        }
                    )
                continue

            # ---- Final answer: stream the text and any client-side tool calls ----
            text = getattr(message, "content", None) or ""
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            if text:
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
            idx += 1

            for c in other_calls:
                tu_block = {
                    "type": "tool_use",
                    "id": c["id"],
                    "name": c["name"],
                    "input": {},
                }
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': tu_block})}\n\n"
                partial = c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"])
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': partial}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
                idx += 1
            if other_calls:
                stop_reason = "tool_use"
            break

        usage = {"output_tokens": output_tokens}
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback

        logger.error(f"Error in streaming web search: {e}\n{traceback.format_exc()}")
        if response_started:
            error_event = {
                "type": "error",
                "error": {"type": "server_error", "message": str(e)},
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "data: [DONE]\n\n"
        else:
            raise


async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    response_started = False  # True once the first SSE byte has been sent
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs

        message_data = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": response_model_name(original_request),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        response_started = True

        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0

        # Process each chunk
        async for chunk in response_generator:
            try:
                # Check if this is the end of the response with usage data
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    if hasattr(chunk.usage, "prompt_tokens"):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, "completion_tokens"):
                        output_tokens = chunk.usage.completion_tokens

                # Handle text content
                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    choice = chunk.choices[0]

                    # Get the delta from the choice
                    if hasattr(choice, "delta"):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, "message", {})

                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, "finish_reason", None)

                    # Process text content
                    delta_content = None

                    # Handle different formats of delta content
                    if hasattr(delta, "content"):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and "content" in delta:
                        delta_content = delta["content"]

                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content

                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"

                    # Process tool calls
                    delta_tool_calls = None

                    # Handle different formats of tool calls
                    if hasattr(delta, "tool_calls"):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and "tool_calls" in delta:
                        delta_tool_calls = delta["tool_calls"]

                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif (
                                accumulated_text
                                and not text_sent
                                and not text_block_closed
                            ):
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]

                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and "index" in tool_call:
                                current_index = tool_call["index"]
                            elif hasattr(tool_call, "index"):
                                current_index = tool_call.index
                            else:
                                current_index = 0

                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index

                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get("function", {})
                                    name = (
                                        function.get("name", "")
                                        if isinstance(function, dict)
                                        else ""
                                    )
                                    tool_id = tool_call.get(
                                        "id", f"toolu_{uuid.uuid4().hex[:24]}"
                                    )
                                else:
                                    function = getattr(tool_call, "function", None)
                                    name = (
                                        getattr(function, "name", "")
                                        if function
                                        else ""
                                    )
                                    tool_id = getattr(
                                        tool_call,
                                        "id",
                                        f"toolu_{uuid.uuid4().hex[:24]}",
                                    )

                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""

                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and "function" in tool_call:
                                function = tool_call.get("function", {})
                                arguments = (
                                    function.get("arguments", "")
                                    if isinstance(function, dict)
                                    else ""
                                )
                            elif hasattr(tool_call, "function"):
                                function = getattr(tool_call, "function", None)
                                arguments = (
                                    getattr(function, "arguments", "")
                                    if function
                                    else ""
                                )

                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, use it
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        # Otherwise, try to parse it
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment, treat it as a string
                                    args_json = arguments

                                # Add to accumulated tool content
                                tool_content += (
                                    args_json if isinstance(args_json, str) else ""
                                )

                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"

                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True

                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"

                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}

                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"

                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue

        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}

            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"

            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = (
            f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        )
        logger.error(error_message)

        if response_started:
            # Headers already sent — emit a proper SSE error event so the client
            # knows the stream ended abnormally rather than hanging indefinitely.
            error_event = {
                "type": "error",
                "error": {"type": "server_error", "message": str(e)},
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "data: [DONE]\n\n"
        else:
            # Nothing sent yet — we can still raise an HTTPException
            raise


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    try:
        # print the body here
        body = await raw_request.body()

        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", "unknown")

        # Preserve the exact model name the client requested so responses echo it
        # back (e.g. "claude-sonnet-4-6") instead of the mapped backend name
        # ("openai/fugu"). The model validator does not persist original_model.
        if original_model and original_model != "unknown":
            request.original_model = original_model

        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/") :]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/") :]

        logger.debug(
            f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}"
        )

        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)

        # Determine which API key to use based on the model
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = OPENAI_API_KEY
            # Use custom OpenAI base URL if configured
            if OPENAI_BASE_URL:
                litellm_request["api_base"] = OPENAI_BASE_URL
                logger.debug(
                    f"Using OpenAI API key and custom base URL {OPENAI_BASE_URL} for model: {request.model}"
                )
            else:
                logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            if USE_VERTEX_AUTH:
                litellm_request["vertex_project"] = VERTEX_PROJECT
                litellm_request["vertex_location"] = VERTEX_LOCATION
                litellm_request["custom_llm_provider"] = "vertex_ai"
                logger.debug(
                    f"Using Gemini ADC with project={VERTEX_PROJECT}, location={VERTEX_LOCATION} and model: {request.model}"
                )
            else:
                litellm_request["api_key"] = GEMINI_API_KEY
                logger.debug(f"Using Gemini API key for model: {request.model}")
        else:
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key for model: {request.model}")

        # For OpenAI models - modify request format to work with limitations
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")

            # For OpenAI models, we need to convert content blocks to simple strings
            # and handle other requirements
            for i, msg in enumerate(litellm_request["messages"]):
                # role:"tool" messages already have string content — leave them alone.
                if msg.get("role") == "tool":
                    continue

                # Assistant messages with tool_calls have content=None by design — keep it.
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # Strip any unsupported keys and continue
                    for key in list(msg.keys()):
                        if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                            del msg[key]
                    continue

                # Fallback: handle message content directly when it's a list of tool_result
                # (should no longer occur after convert_anthropic_to_litellm fixes, but kept
                # as a safety net for unexpected shapes)
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if (
                            not isinstance(block, dict)
                            or block.get("type") != "tool_result"
                        ):
                            is_only_tool_result = False
                            break

                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(
                            f"Found message with only tool_result content - special handling required"
                        )
                        # Extract the content from all tool_result blocks
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])

                            # Handle different formats of content
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if (
                                        isinstance(item, dict)
                                        and item.get("type") == "text"
                                    ):
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # Fall back to string representation of any dict
                                        try:
                                            item_text = item.get(
                                                "text", json.dumps(item)
                                            )
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"

                        # Replace the list with extracted text
                        litellm_request["messages"][i]["content"] = (
                            all_text.strip() or "..."
                        )
                        logger.warning(
                            f"Converted tool_result to plain text: {all_text.strip()[:200]}..."
                        )
                        continue  # Skip normal processing for this message

                # 1. Handle content field - normal case
                if "content" in msg:
                    # Check if content is a list (content blocks)
                    if isinstance(msg["content"], list):
                        # Convert complex content blocks to simple string
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # Handle different content block types
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"

                                # Handle tool_result content blocks - extract nested text
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"

                                    # Extract text from the tool_result content
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if (
                                                isinstance(item, dict)
                                                and item.get("type") == "text"
                                            ):
                                                text_content += (
                                                    item.get("text", "") + "\n"
                                                )
                                            elif isinstance(item, dict):
                                                # Handle any dict by trying to extract text or convert to JSON
                                                if "text" in item:
                                                    text_content += (
                                                        item.get("text", "") + "\n"
                                                    )
                                                else:
                                                    try:
                                                        text_content += (
                                                            json.dumps(item) + "\n"
                                                        )
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # Handle dictionary content
                                        if result_content.get("type") == "text":
                                            text_content += (
                                                result_content.get("text", "") + "\n"
                                            )
                                        else:
                                            try:
                                                text_content += (
                                                    json.dumps(result_content) + "\n"
                                                )
                                            except:
                                                text_content += (
                                                    str(result_content) + "\n"
                                                )
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += (
                                                json.dumps(result_content) + "\n"
                                            )
                                        except:
                                            text_content += str(result_content) + "\n"

                                # Handle tool_use content blocks
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"

                                # Handle image content blocks
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"

                        # Make sure content is never empty for OpenAI models
                        if not text_content.strip():
                            text_content = "..."

                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # Also check for None or empty string content
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = (
                            "..."  # Empty content not allowed
                        )

                # 2. Remove any fields OpenAI doesn't support in messages
                for key in list(msg.keys()):
                    if key not in [
                        "role",
                        "content",
                        "name",
                        "tool_call_id",
                        "tool_calls",
                    ]:
                        logger.warning(
                            f"Removing unsupported field from message: {key}"
                        )
                        del msg[key]

            # 3. Final validation - check for any remaining invalid values and dump full message details
            for i, msg in enumerate(litellm_request["messages"]):
                # Log the message format for debugging
                logger.debug(
                    f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}"
                )

                # If content is still a list or None, replace with placeholder
                if isinstance(msg.get("content"), list):
                    logger.warning(
                        f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}"
                    )
                    # Last resort - stringify the entire content as JSON
                    litellm_request["messages"][i]["content"] = (
                        f"Content as JSON: {json.dumps(msg.get('content'))}"
                    )
                elif msg.get("content") is None and not msg.get("tool_calls"):
                    # None content is valid for assistant messages that have tool_calls;
                    # only replace it for other messages.
                    logger.warning(
                        f"Message {i} has None content - replacing with placeholder"
                    )
                    litellm_request["messages"][i]["content"] = (
                        "..."  # Fallback placeholder
                    )

        # Only log basic info about the request, not the full details
        logger.debug(
            f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}"
        )

        # --- Web search path -------------------------------------------------
        # If the client requested Anthropic's web_search tool, run it ourselves
        # via Exa (the backend cannot perform server-side search). This drives an
        # agentic loop, executes searches, and returns the final answer with
        # web_search_tool_result blocks.
        web_search_requested = bool(request.tools) and any(
            t.is_web_search() for t in request.tools
        )
        if web_search_requested:
            allowed_domains = None
            blocked_domains = None
            for t in request.tools:
                if t.is_web_search():
                    allowed_domains = getattr(t, "allowed_domains", None)
                    blocked_domains = getattr(t, "blocked_domains", None)
                    break

            num_tools = len(request.tools) if request.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get("model"),
                len(litellm_request["messages"]),
                num_tools,
                200,
            )

            # Streaming: run the loop *inside* the stream so the live
            # WebSearch("…") status renders as each search happens.
            if request.stream:
                return StreamingResponse(
                    handle_streaming_web_search(
                        litellm_request, request, allowed_domains, blocked_domains
                    ),
                    media_type="text/event-stream",
                )

            # Non-streaming: run the loop and return all blocks at once.
            final_response, search_records = await retry_with_backoff(
                lambda: asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: run_web_search_loop_sync(
                        litellm_request, allowed_domains, blocked_domains
                    ),
                )
            )
            logger.debug(
                f"🔎 WEB SEARCH COMPLETE: {len(search_records)} search(es) executed"
            )

            anthropic_response = convert_litellm_to_anthropic(
                final_response, request, search_records
            )
            return anthropic_response

        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get("model"),
                len(litellm_request["messages"]),
                num_tools,
                200,  # Assuming success at this point
            )
            # Ensure we use the async version for streaming; retry on transient errors
            response_generator = await retry_with_backoff(
                lambda: litellm.acompletion(**litellm_request)
            )

            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream",
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get("model"),
                len(litellm_request["messages"]),
                num_tools,
                200,  # Assuming success at this point
            )
            start_time = time.time()
            litellm_response = await retry_with_backoff(
                lambda: asyncio.get_event_loop().run_in_executor(
                    None, lambda: litellm.completion(**litellm_request)
                )
            )
            logger.debug(
                f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s"
            )

            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)

            return anthropic_response

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()

        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback,
        }

        # Check for LiteLLM-specific attributes
        for attr in ["message", "status_code", "response", "llm_provider", "model"]:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)

        # Check for additional exception details in dictionaries
        if hasattr(e, "__dict__"):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ["args", "__traceback__"]:
                    error_details[key] = str(value)

        # Helper function to safely serialize objects for JSON
        def sanitize_for_json(obj):
            """递归地清理对象使其可以JSON序列化"""
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif hasattr(obj, "__dict__"):
                return sanitize_for_json(obj.__dict__)
            elif hasattr(obj, "text"):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

        # Log all error details with safe serialization
        sanitized_details = sanitize_for_json(error_details)
        logger.error(
            f"Error processing request: {json.dumps(sanitized_details, indent=2)}"
        )

        # Log a compact summary of the *request* that failed. This is what lets us
        # tell a classifier/Edit failure apart from a normal one, and spot bad
        # message structures (e.g. ends-with-assistant, empty content, huge size).
        try:
            roles = [m.role for m in request.messages]
            sys_len = 0
            if isinstance(request.system, str):
                sys_len = len(request.system)
            elif isinstance(request.system, list):
                sys_len = sum(len(getattr(b, "text", "") or "") for b in request.system)
            logger.error(
                "Failed request shape: model=%s stream=%s max_tokens=%s "
                "messages=%d roles=%s tools=%d tool_choice=%s system_chars=%d"
                % (
                    request.model,
                    request.stream,
                    request.max_tokens,
                    len(request.messages),
                    "".join("u" if r == "user" else "a" for r in roles),
                    len(request.tools) if request.tools else 0,
                    json.dumps(request.tool_choice) if request.tool_choice else "none",
                    sys_len,
                )
            )
        except Exception as shape_err:
            logger.error(f"(could not log request shape: {shape_err})")

        # Format error for response
        error_message = f"Error: {str(e)}"
        if "message" in error_details and error_details["message"]:
            error_message += f"\nMessage: {error_details['message']}"
        if "response" in error_details and error_details["response"]:
            error_message += f"\nResponse: {error_details['response']}"

        # Return detailed error
        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message)


def _approx_token_count(messages: List[Dict[str, Any]]) -> int:
    """Rough offline token estimate (~4 chars/token) for count_tokens fallbacks.

    Used when litellm.token_counter() is unavailable or raises for unknown/custom
    models (e.g. Sakana's fugu). Claude Code only needs an approximate number.
    """
    total_chars = 0
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if isinstance(block.get("text"), str):
                        total_chars += len(block["text"])
                    else:
                        total_chars += len(json.dumps(block))
                else:
                    total_chars += len(str(block))
        elif content is not None:
            total_chars += len(str(content))
    # ~4 characters per token; never report 0 for a non-empty request.
    return max(1, total_chars // 4)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    try:
        # Recover the exact model name the client sent (the validator maps
        # request.model to the backend name and does not persist original_model).
        try:
            body_json = json.loads((await raw_request.body()).decode("utf-8"))
            original_model = body_json.get("model") or request.model
        except Exception:
            original_model = request.original_model or request.model
        request.original_model = original_model

        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/") :]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/") :]

        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking,
            )
        )

        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter

            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get("model"),
                len(converted_request["messages"]),
                num_tools,
                200,  # Assuming success at this point
            )

            # Prepare token counter arguments
            token_counter_args = {
                "model": converted_request["model"],
                "messages": converted_request["messages"],
            }
            # NOTE: litellm.token_counter() counts tokens locally (offline) and
            # does NOT accept network args like `api_base`/`api_key`. Passing them
            # raises `TypeError: unexpected keyword argument 'api_base'`.

            # Count tokens
            token_count = token_counter(**token_counter_args)

            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)

        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(
                input_tokens=_approx_token_count(converted_request["messages"])
            )
        except Exception as e:
            # token_counter can fail for unknown/custom models (e.g. fugu) or on
            # signature mismatches across litellm versions. Don't 500 the client —
            # Claude Code only needs a rough count, so fall back to an estimate.
            logger.warning(
                f"token_counter failed ({e}); using char-based approximation"
            )
            return TokenCountResponse(
                input_tokens=_approx_token_count(converted_request["messages"])
            )

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}


# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"


def log_request_beautifully(
    method, path, claude_model, openai_model, num_messages, num_tools, status_code
):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"

    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"

    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

    # Format status code
    status_str = (
        f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}"
        if status_code == 200
        else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    )

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"

    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
