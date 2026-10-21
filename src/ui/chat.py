"""
Chat UI: message rendering, streaming, and agent invocation.
"""
import streamlit as st

from src.agent.executor import build_executor
from src.ui.rate_limit import (
    get_remaining_queries,
    increment_query_count,
    is_rate_limited,
)


def init_chat():
    """Initialize the messages list if not present."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def render_chat_history():
    """Render all previous messages in the session."""
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _render_tool_trace(tool_calls):
    """
    Render the tools that were called, collapsed by default.

    tool_calls: list of dicts with keys 'name' and 'arguments'
    """
    if not tool_calls:
        return
    with st.expander(f"🔧 {len(tool_calls)} tool call(s)", expanded=False):
        for i, call in enumerate(tool_calls, start=1):
            st.markdown(f"**{i}. `{call['name']}`**")
            st.code(str(call.get("arguments", {})), language="python")


def _extract_tool_calls(agent_messages) -> list[dict]:
    """
    Extract tool-call info from the agent's message list.

    LangChain v1 agent returns AIMessage objects with a `.tool_calls` attribute
    when the LLM decides to call a tool.
    """
    tool_calls = []
    for msg in agent_messages:
        calls = getattr(msg, "tool_calls", None)
        if calls:
            for c in calls:
                tool_calls.append({
                    "name": c.get("name", "unknown"),
                    "arguments": c.get("args", {}),
                })
    return tool_calls


def stream_agent_response(user_input: str, mode: str):
    """
    Invoke the agent with streaming and render the response in the chat.

    Streamlit's st.write_stream consumes an iterator of text chunks and
    renders them incrementally. We split the work into two passes:

    Pass 1 (non-streaming): Run the agent to get the full result, including
                             tool call traces and final message.
    Pass 2 (streaming):     Simulate streaming by yielding the final message
                             in chunks — because the LangChain v1 agent
                             doesn't expose tool call traces while streaming
                             at token level cleanly.

    This hybrid approach gives the user the "tokens appear" experience
    without losing the tool-call transparency.
    """
    executor = build_executor(mode)

    with st.spinner("Thinking... (the agent may call tools)"):
        result = executor.invoke({"messages": [{"role": "user", "content": user_input}]})

    # Extract tool calls for transparency
    tool_calls = _extract_tool_calls(result["messages"])

    # Extract final answer text
    final_msg = result["messages"][-1]
    final_text = getattr(final_msg, "content", str(final_msg))

    # Render tool trace below the answer (collapsed)
    def chunked_stream(text, chunk_size=8):
        """Simulate token-by-token streaming of the final answer."""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    return final_text, tool_calls, chunked_stream(final_text)


def handle_chat_input(mode: str):
    """
    Handle the chat input box and agent invocation.

    Returns nothing; renders directly to the Streamlit app.
    """
    user_input = st.chat_input("Ask about solar forecasts, EV loads, or historical data...")

    if user_input is None:
        return

    # Enforce rate limit
    if is_rate_limited():
        st.error("Query limit reached. Refresh the page to start a new session.")
        return

    # Append user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent with streaming
    with st.chat_message("assistant"):
        final_text, tool_calls, stream = stream_agent_response(user_input, mode)
        rendered = st.write_stream(stream)
        if tool_calls:
            _render_tool_trace(tool_calls)

    # Persist assistant message
    st.session_state["messages"].append({
        "role": "assistant",
        "content": rendered,
    })

    # Increment rate limit AFTER successful query
    increment_query_count()

    # Show remaining budget in a small toast (auto-dismisses)
    remaining = get_remaining_queries()
    if remaining <= 5:
        st.toast(f"⚠️ {remaining} queries remaining this session", icon="⚠️")