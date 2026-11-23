"""Default teacher identity for the reference implementation."""
import os

MODEL = os.environ.get("PRIORA_MODEL", "gpt-6.2-astra")
EFFORT = os.environ.get("PRIORA_EFFORT", "xhigh")
PROVIDER = "priora_endpoint" if os.getenv("PRIORA_BASE_URL") else "openai"
DEFAULT_CODEX_IMAGE_MAX_EDGE = 480
