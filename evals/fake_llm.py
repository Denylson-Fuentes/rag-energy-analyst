"""
A scripted chat model for testing the evaluation harness without an LLM.

It calls the oracle tools stored in the eval set, then answers by echoing the
tool output. Results produced with it measure the harness and the tool layer,
NOT a language model, and are never reported as model results.
"""
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedToolModel(BaseChatModel):
    script: dict  # question text -> list of {"tool", "args"}

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            calls = self.script.get(last.content) or [
                {"tool": "search_energy_data", "args": {"query": last.content}}
            ]
            msg = AIMessage(content="", tool_calls=[
                {"name": c["tool"], "args": c["args"], "id": f"call_{uuid.uuid4().hex[:8]}"} for c in calls
            ])
        else:
            outs = [m.content for m in messages if isinstance(m, ToolMessage)]
            msg = AIMessage(content="\n\n".join(outs))
        return ChatResult(generations=[ChatGeneration(message=msg)])
