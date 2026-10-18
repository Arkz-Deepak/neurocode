"""
Executes the comparative benchmark and outputs the empirical table for the research paper.
"""

import os
import tempfile
from pathlib import Path
from src.benchmark.evaluator import PipelineEvaluator


def create_benchmark_repo(base_dir: Path):
    """Creates a realistic repository with 8 distinct modules and cross-file dependencies."""
    modules = {
        "core/auth.py": """
from core.database import get_user, save_token
from utils.logger import log_event

def login_user(username, password):
    user = get_user(username)
    if not user or user['password'] != password:
        log_event(f"Failed login for {username}")
        return None
    token = f"tok_{username}_123"
    save_token(username, token)
    return token
""",
        "core/database.py": """
DB = {"alice": {"password": "secret", "role": "admin"}}
TOKENS = {}

def get_user(username):
    return DB.get(username)

def save_token(username, token):
    TOKENS[username] = token
""",
        "utils/logger.py": """
def log_event(message):
    print(f"[LOG] {message}")
""",
        "models/dataset.py": """
class DatasetLoader:
    def __init__(self, path):
        self.path = path
    def load(self):
        return [1, 2, 3, 4, 5]
""",
        "models/trainer.py": """
from models.dataset import DatasetLoader
class ModelTrainer:
    def __init__(self, data_path):
        self.loader = DatasetLoader(data_path)
    def train(self):
        data = self.loader.load()
        return f"Trained on {len(data)} items"
""",
        "math_lib/matrix.py": """
def matrix_multiply(a, b):
    return [[sum(i * j for i, j in zip(r, c)) for c in zip(*b)] for r in a]
""",
        "math_lib/solver.py": """
from math_lib.matrix import matrix_multiply
def solve_linear_system(a, b):
    return matrix_multiply(a, b)
""",
        "frontend/cli.py": """
from core.auth import login_user
def run_cli():
    user = input("User: ")
    pwd = input("Pass: ")
    tok = login_user(user, pwd)
    print("Logged in!" if tok else "Denied!")
"""
    }

    for rel_path, code in modules.items():
        p = base_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code.strip())


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir)
        create_benchmark_repo(repo_path)

        evaluator = PipelineEvaluator(str(repo_path))

        issue = "Security issue: login_user in core/auth.py logs plaintext username in log_event and doesn't handle empty password."
        target = "login_user"

        print("\n==========================================================================")
        print("          NEUROCODE RESEARCH BENCHMARK - HOST-CPU VS BASELINES           ")
        print("==========================================================================\n")

        res_raw = evaluator.run_raw_dump(issue, target)
        res_bm25 = evaluator.run_bm25_rag(issue, target)
        res_neuro = evaluator.run_neurocode(issue, target)

        results = [res_raw, res_bm25, res_neuro]

        # Markdown Table for IEEE Paper / Kaggle Writeup
        print("| Method / Pipeline | Prompt Tokens | Token Reduction | CPU Preprocessing Time | Est. KV-Cache VRAM | Target Captured? |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- |")

        baseline_tokens = res_raw.prompt_tokens
        for r in results:
            reduction = f"-{((baseline_tokens - r.prompt_tokens) / baseline_tokens) * 100:.1f}%" if r.prompt_tokens < baseline_tokens else "Baseline (0%)"
            captured = "PASS" if r.captured_target_entity else "FAIL"
            print(f"| **{r.method_name}** | {r.prompt_tokens:,} | **{reduction}** | {r.preprocessing_time_ms:.2f} ms | {r.estimated_kv_vram_mb:.1f} MB | {captured} |")

        print("\n==========================================================================")
        token_savings = ((res_raw.prompt_tokens - res_neuro.prompt_tokens) / res_raw.prompt_tokens) * 100
        print(f"SUMMARY: NeuroCode achieved {token_savings:.1f}% token reduction while running in {res_neuro.preprocessing_time_ms:.2f} ms on Host CPU!")
        print("==========================================================================\n")


if __name__ == "__main__":
    main()
