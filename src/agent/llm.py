"""
LLM factory. Production runs Claude Haiku 4.5 on Amazon Bedrock; the eval
harness can also run against the Claude API (LLM_PROVIDER=anthropic).
"""
import os

from src import config

_llm = {}


def _load_streamlit_aws_secrets():
    """On Streamlit Cloud, AWS credentials live in st.secrets; copy them to env."""
    try:
        import streamlit as st

        for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = st.secrets[key]
    except Exception:
        pass


def get_llm(provider: str | None = None):
    provider = provider or config.LLM_PROVIDER
    if provider in _llm:
        return _llm[provider]

    if provider == "bedrock":
        from langchain_aws import ChatBedrock

        _load_streamlit_aws_secrets()
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        llm = ChatBedrock(
            model_id=config.BEDROCK_MODEL_ID,
            region_name=region,
            model_kwargs={"temperature": 0, "max_tokens": 1024},
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=config.ANTHROPIC_MODEL_ID, max_tokens=1024, max_retries=4)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r} (use 'bedrock' or 'anthropic')")

    _llm[provider] = llm
    return llm
