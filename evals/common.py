import json
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(EVAL_DIR, "data", "questions.jsonl")
RESULTS_DIR = os.getenv("EVAL_RESULTS_DIR", os.path.join(EVAL_DIR, "..", "reports", "results"))


def load_questions(path: str = QUESTIONS) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def results_path(name: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


def write_json(name: str, obj) -> str:
    p = results_path(name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return p


def write_jsonl(name: str, rows) -> str:
    p = results_path(name)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return p


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
