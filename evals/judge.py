"""
Optional LLM-as-judge for groundedness (cross-check of the deterministic
numeric scorer; it also catches non-numeric unsupported claims such as
invented explanations).

Judge model is configured independently from the agent so the agent never
grades itself:
    JUDGE_PROVIDER=anthropic JUDGE_MODEL=claude-opus-5-5
    JUDGE_PROVIDER=bedrock   JUDGE_MODEL=<bedrock model or inference-profile id>
"""
import os

from pydantic import BaseModel, Field

JUDGE_PROMPT = """You are auditing an AI assistant's answer for groundedness.

EVIDENCE (the only information the assistant had, from its tool calls):
<evidence>
{evidence}
</evidence>

QUESTION: {question}

ANSWER:
<answer>
{answer}
</answer>

Split the answer into atomic factual claims (numbers, dates, node names,
comparisons, explanations). Ignore statements that only say data is
unavailable, and ignore restatements of the question. For each claim decide
whether the evidence supports it. Arithmetic on evidence values (differences,
sums, percentages, rounding) counts as supported."""


class Claim(BaseModel):
    claim: str
    supported: bool


class Verdict(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


def get_judge():
    provider = os.getenv("JUDGE_PROVIDER", "anthropic")
    model = os.getenv("JUDGE_MODEL", "claude-opus-5-5")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, max_tokens=4096, max_retries=4)
    elif provider == "bedrock":
        from langchain_aws import ChatBedrock

        llm = ChatBedrock(model_id=model, region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                          model_kwargs={"max_tokens": 4096})
    else:
        raise ValueError(provider)
    return llm.with_structured_output(Verdict)


def judge_record(judge, question: str, answer: str, tool_outputs: list[str]) -> dict:
    evidence = "\n\n---\n\n".join(tool_outputs)[:30000] or "(no tool calls were made)"
    v: Verdict = judge.invoke(JUDGE_PROMPT.format(evidence=evidence, question=question, answer=answer))
    n = len(v.claims)
    return {
        "n_claims": n,
        "n_supported": sum(c.supported for c in v.claims),
        "unsupported": [c.claim for c in v.claims if not c.supported],
    }
