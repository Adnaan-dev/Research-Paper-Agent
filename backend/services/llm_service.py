"""
llm_service.py — Central LLM factory.

Returns a LangChain-compatible ChatModel based on LLM_PROVIDER in .env:

  "groq"   → FREE. Groq cloud API with llama-3.3-70b-versatile.
             Get your free API key at: https://console.groq.com
             Groq's free tier: 14,400 requests/day, 6000 tokens/min

  "openai" → Paid. OpenAI GPT-4o-mini. Requires billing credits.

Usage in agents:
    from services.llm_service import get_llm
    llm = get_llm(temperature=0.3)
"""

from loguru import logger
from config import get_settings


def get_llm(temperature: float = 0.3, max_tokens: int = 2048):
    """
    Return the configured LangChain chat model.

    Args:
        temperature: 0.0 = deterministic, 1.0 = creative
        max_tokens:  Maximum tokens in the response

    Returns:
        A LangChain BaseChatModel (Groq or OpenAI)
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "groq":
        return _get_groq_llm(temperature, max_tokens)
    elif provider == "openai":
        return _get_openai_llm(temperature, max_tokens)
    else:
        logger.warning(f"[LLM] Unknown provider '{provider}' — defaulting to Groq")
        return _get_groq_llm(temperature, max_tokens)


def _get_groq_llm(temperature: float, max_tokens: int):
    """Initialize Groq LLM (FREE tier available)."""
    settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set in your .env file!\n"
            "Get a FREE key at https://console.groq.com\n"
            "Then add: GROQ_API_KEY=gsk_your_key_here"
        )

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError(
            "langchain-groq is not installed. Run:\n"
            "  pip install langchain-groq"
        )

    logger.info(f"[LLM] Using Groq: {settings.groq_model} (temp={temperature})")
    return ChatGroq(
        model=settings.groq_model,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=settings.groq_api_key,
        # Groq has generous rate limits but add a small stop-gap
        request_timeout=60,
    )


def _get_openai_llm(temperature: float, max_tokens: int):
    """Initialize OpenAI LLM (requires billing credits)."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set in your .env file!\n"
            "Add credits at https://platform.openai.com/billing\n"
            "Or switch to Groq (free): set LLM_PROVIDER=groq in .env"
        )

    from langchain_openai import ChatOpenAI
    logger.info(f"[LLM] Using OpenAI: {settings.openai_model} (temp={temperature})")
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=settings.openai_api_key,
        request_timeout=120,
    )
