"""
CLI test for the agent — LangChain v1.
"""
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.agent.executor import build_executor


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "operator"
    print(f"Building {mode} agent...\n")
    agent = build_executor(mode)
    print("Agent ready. Type 'exit' to quit.\n")

    while True:
        question = input("> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            # v1 returns a dict with 'messages'; last message is the answer
            final = result["messages"][-1]
            print(f"\n{final.content}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()