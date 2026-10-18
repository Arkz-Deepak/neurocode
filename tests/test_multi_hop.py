"""
Demonstrates the Core Novelty Failure Mode:
Where BM25 Lexical RAG (Agentless) FAILS to capture indirect dependencies,
while NeuroCode Host-CPU Slicer SUCCEEDS via topological AST call-graph reachability.
"""

from pathlib import Path
import tempfile
from src.benchmark.evaluator import PipelineEvaluator


def test_indirect_dependency_resolution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir)
        api_dir = repo_path / "api"
        crypto_dir = repo_path / "crypto"
        api_dir.mkdir()
        crypto_dir.mkdir()

        # Caller file (contains the issue keywords, but NOT the bug)
        (api_dir / "gateway.py").write_text("""
from crypto.signer import verify_signature

def handle_incoming_request(request_payload):
    '''Entrypoint that fails when client signature is rejected.'''
    sig = request_payload.get('signature')
    data = request_payload.get('body')
    if not verify_signature(data, sig):
        raise ValueError("Invalid client signature verification failed")
    return {"status": "success"}
""")

        # Callee file where the ACTUAL BUG lives (DOES NOT contain words like 'incoming', 'request', or 'client')
        (crypto_dir / "signer.py").write_text("""
import hmac
import hashlib

def verify_signature(data: str, signature: str) -> bool:
    '''Calculates HMAC SHA256 digest.'''
    # BUG: Doesn't handle none/empty string
    expected = hmac.new(b'secret', data.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
""")

        # Unrelated distractor file filled with keyword matches like 'client', 'signature', 'request'
        (api_dir / "distractor.py").write_text("""
def client_request_logger(client_name, signature_status):
    '''Logs client request signature verification status.'''
    print(f"Client {client_name} request signature was {signature_status}")
""")

        evaluator = PipelineEvaluator(str(repo_path))

        # Issue mentions gateway and client signature failure, but NOT the crypto helper
        issue = "Error: handle_incoming_request in api/gateway.py crashes with Invalid client signature verification failed on empty payloads."
        # Target implementation to capture
        target_bug_symbol = "def verify_signature"

        res_bm25 = evaluator.run_bm25_rag(issue, target_bug_symbol)
        res_neuro = evaluator.run_neurocode(issue, target_bug_symbol)

        print("\n==========================================================================")
        print("          MULTI-HOP BUG LOCALIZATION BENCHMARK: BM25 VS NEUROCODE        ")
        print("==========================================================================")
        print(f"Target Buggy Function Implementation to Capture: `{target_bug_symbol}` in crypto/signer.py\n")
        print(f"1. BM25 Lexical Search (Agentless Baseline): Captured Implementation? -> {'PASS' if res_bm25.captured_target_entity else 'FAIL (Missed root cause implementation!)'}")
        print(f"2. NeuroCode AST Slicer (Our Method):        Captured Implementation? -> {'PASS (Followed AST call edge to implementation!)' if res_neuro.captured_target_entity else 'FAIL'}")
        print("==========================================================================\n")

        # BM25 is deceived by the distractor file with matching keywords and misses the definition in crypto/signer.py
        assert not res_bm25.captured_target_entity, "BM25 should have missed the indirect implementation!"
        # NeuroCode traces the AST call edge from handle_incoming_request -> def verify_signature
        assert res_neuro.captured_target_entity, "NeuroCode must capture the indirect callee implementation!"
