"""
Unit tests and latency benchmarking for NeuroCode Host-CPU Graph Slicer.
"""

import os
import tempfile
import time
from pathlib import Path
import pytest
from src.graph.builder import RepoGraphBuilder
from src.graph.slicer import CodeGraphSlicer


@pytest.fixture
def sample_repo(tmp_path):
    """Creates a realistic multi-file repository structure with related and unrelated modules."""
    auth_dir = tmp_path / "auth"
    notify_dir = tmp_path / "notify"
    unrelated_dir = tmp_path / "unrelated"

    auth_dir.mkdir()
    notify_dir.mkdir()
    unrelated_dir.mkdir()

    # File 1: auth/user.py
    (auth_dir / "user.py").write_text("""
class User:
    '''Represents a system user.'''
    def __init__(self, username: str):
        self.username = username

def validate_password(password: str) -> bool:
    '''Validates user password strength.'''
    if len(password) < 8:
        return False
    return True
""")

    # File 2: auth/login.py
    (auth_dir / "login.py").write_text("""
from auth.user import User, validate_password
from notify.email import send_alert

def authenticate(username: str, password: str) -> bool:
    '''Authenticates a user and dispatches alert on failure.'''
    if not validate_password(password):
        send_alert(username, "Failed login attempt")
        return False
    return True
""")

    # File 3: notify/email.py
    (notify_dir / "email.py").write_text("""
def send_alert(user: str, message: str) -> None:
    '''Dispatches an email notification.'''
    print(f"Alert to {user}: {message}")
""")

    # File 4: Unrelated massive file
    (unrelated_dir / "math_utils.py").write_text("""
def calculate_pi_digits(n: int) -> float:
    '''Unrelated heavy utility.'''
    return 3.14159

def compute_fibonacci(n: int) -> int:
    '''Unrelated calculation.'''
    return n if n <= 1 else compute_fibonacci(n-1) + compute_fibonacci(n-2)
""")

    return tmp_path


def test_graph_builder_and_slicing_precision(sample_repo):
    start_time = time.perf_counter()

    # 1. Build in-memory graph
    builder = RepoGraphBuilder(str(sample_repo))
    graph = builder.build()
    build_time_ms = (time.perf_counter() - start_time) * 1000

    assert graph.number_of_nodes() > 0
    assert build_time_ms < 100.0, f"Graph build was too slow: {build_time_ms:.2f}ms"

    # 2. Slice context for target issue
    issue = "Error in authenticate function: validate_password fails with empty strings."
    slicer = CodeGraphSlicer(graph)
    seeds = slicer.find_seed_nodes(issue)

    assert any("authenticate" in s for s in seeds), "Seed node 'authenticate' was not localized!"

    subgraph = slicer.extract_subgraph(seeds, k_hops=1)
    slice_text = slicer.format_slice_for_llm(subgraph, seeds)

    # 3. Assertions on precision and compression
    assert "def authenticate" in slice_text
    assert "validate_password" in slice_text
    # Unrelated functions must be completely pruned
    assert "calculate_pi_digits" not in slice_text
    assert "compute_fibonacci" not in slice_text

    print(f"\n[BENCHMARK] Host CPU Graph Build & Slicing Latency: {build_time_ms:.2f} ms")
    print(f"[BENCHMARK] Extracted Context Size: {len(slice_text.split())} words (~{int(len(slice_text.split()) * 1.3)} tokens)")
    print("\n--- GENERATED SLICE PROMPT ---\n")
    print(slice_text)
