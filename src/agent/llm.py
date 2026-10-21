import os
from langchain_aws import ChatBedrock


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        # Try Streamlit secrets first (deployed), then env vars (local)
        try:
            import streamlit as st
            aws_key = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
            aws_secret = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
            aws_region = st.secrets.get("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        except Exception:
            aws_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        # ChatBedrock reads AWS creds from boto3 session or env vars
        # So we set them into the environment if not already there
        if aws_key:
            os.environ["AWS_ACCESS_KEY_ID"] = aws_key
        if aws_secret:
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret
        if aws_region:
            os.environ["AWS_DEFAULT_REGION"] = aws_region

        _llm = ChatBedrock(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name=aws_region or "us-east-1",
            model_kwargs={"temperature": 0, "max_tokens": 1024},
        )
    return _llm