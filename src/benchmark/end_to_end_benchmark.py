"""
NeuroCode End-to-End Bug Resolution Benchmark Harness.
Directly addresses Academic Review Gap 1:
Evaluates real repository repair with automated test suites (pytest),
comparing:
  1. Raw Context Dump (Naive Baseline)
  2. BM25 Lexical RAG (Agentless Baseline)
  3. Unconstrained Graph Traversal (RepoGraph Baseline)
  4. NeuroCode Priority-Ranked Knapsack Slicer (Our Proposed Method)

Measures:
  - Retrieval Accuracy (Target Bug Entity captured)
  - Patch Generation & Syntactic Validity
  - Unit Test Pass Rate (% Resolved)
  - Token Footprint & Peak GPU VRAM
  - Preprocessing, Inference, and Total Latencies
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.graph.builder import RepoGraphBuilder
from src.graph.slicer import CodeGraphSlicer
from src.agent.patcher import PatchApplier, SYSTEM_PROMPT


@dataclass
class BugTask:
    task_id: str
    title: str
    problem_statement: str
    files: Dict[str, str]
    test_path: str
    test_code: str
    target_bug_file: str
    target_bug_symbol: str
    category: str


@dataclass
class MethodEvaluationResult:
    method: str
    task_id: str
    input_tokens: int
    output_tokens: int
    prep_time_ms: float
    infer_time_ms: float
    total_time_ms: float
    peak_vram_mb: float
    captured_target: bool
    valid_patch_format: bool
    patch_applied: bool
    test_passed: bool
    model_output: str
    error_message: Optional[str] = None


def get_curated_benchmark_tasks() -> List[BugTask]:
    """
    Curated suite of realistic multi-module software engineering bugs.
    Each bug has a multi-file dependency structure and an isolated unit test suite.
    """
    tasks = []

    # -------------------------------------------------------------------------
    # Task 1: Cross-File Indirect Call Bug
    # Issue keywords describe API gateway crash, but root cause is in crypto signer.
    # -------------------------------------------------------------------------
    tasks.append(BugTask(
        task_id="NC-TASK-001",
        title="Indirect Callee NoneType Crash in HMAC Verification",
        category="Indirect Call Chain",
        problem_statement=(
            "Error: handle_incoming_request in api/gateway.py crashes with AttributeError "
            "when client signature payload contains None body. Client request authentication "
            "must return False instead of raising an unhandled exception."
        ),
        files={
            "api/gateway.py": (
                "from crypto.signer import verify_signature\n\n"
                "def handle_incoming_request(request_payload: dict) -> dict:\n"
                "    '''Entrypoint that processes incoming client requests.'''\n"
                "    sig = request_payload.get('signature')\n"
                "    data = request_payload.get('body')\n"
                "    if not verify_signature(data, sig):\n"
                "        raise ValueError('Invalid client signature verification failed')\n"
                "    return {'status': 'success'}\n"
            ),
            "crypto/signer.py": (
                "import hmac\n"
                "import hashlib\n\n"
                "def verify_signature(data: str, signature: str) -> bool:\n"
                "    '''Calculates HMAC SHA256 digest.'''\n"
                "    if not signature:\n"
                "        return False\n"
                "    expected = hmac.new(b'secret_key', data.encode('utf-8'), hashlib.sha256).hexdigest()\n"
                "    return hmac.compare_digest(expected, signature)\n"
            ),
            "api/logger.py": (
                "def client_request_logger(client_name: str, signature_status: str):\n"
                "    '''Logs client request signature verification status.'''\n"
                "    print(f'Client {client_name} signature: {signature_status}')\n"
            ),
            "common/utils.py": (
                "def format_response(code: int, message: str) -> dict:\n"
                "    return {'code': code, 'message': message}\n"
            ),
        },
        test_path="tests/test_gateway.py",
        test_code=(
            "import pytest\n"
            "from api.gateway import handle_incoming_request\n"
            "from crypto.signer import verify_signature\n\n"
            "def test_signature_handles_none():\n"
            "    # Fails if verify_signature crashes on data=None\n"
            "    res = verify_signature(None, 'some_sig')\n"
            "    assert res is False\n\n"
            "def test_gateway_handles_none_body():\n"
            "    with pytest.raises(ValueError, match='Invalid client signature'):\n"
            "        handle_incoming_request({'body': None, 'signature': 'abc'})\n"
        ),
        target_bug_file="crypto/signer.py",
        target_bug_symbol="def verify_signature",
    ))

    # -------------------------------------------------------------------------
    # Task 2: Class Inheritance Hierarchy Override Bug
    # Subclass handler crashes because parent BaseSerializer lacks ISO-8601 parsing.
    # -------------------------------------------------------------------------
    tasks.append(BugTask(
        task_id="NC-TASK-002",
        title="Class Inheritance Fallback in Datetime Serialization",
        category="Inheritance Hierarchy",
        problem_statement=(
            "TypeError: CustomJsonSerializer in handlers/json_handler.py fails when serializing "
            "objects containing datetime instances. Base serializer fallback must support "
            "ISO-8601 formatting for datetime objects."
        ),
        files={
            "core/base_handler.py": (
                "from datetime import datetime\n\n"
                "class BaseSerializer:\n"
                "    '''Base serializer with primitive dispatch.'''\n"
                "    def serialize_value(self, val):\n"
                "        if isinstance(val, (int, float, str, bool)):\n"
                "            return val\n"
                "        raise TypeError(f'Type {type(val)} not serializable')\n"
            ),
            "handlers/json_handler.py": (
                "import json\n"
                "from core.base_handler import BaseSerializer\n\n"
                "class CustomJsonSerializer(BaseSerializer):\n"
                "    '''JSON serializer extending base serialization.'''\n"
                "    def serialize(self, data: dict) -> str:\n"
                "        out = {}\n"
                "        for k, v in data.items():\n"
                "            out[k] = self.serialize_value(v)\n"
                "        return json.dumps(out)\n"
            ),
            "handlers/xml_handler.py": (
                "class XmlSerializer:\n"
                "    def serialize(self, data):\n"
                "        return '<xml></xml>'\n"
            ),
            "common/helpers.py": (
                "def dump_json(obj):\n"
                "    return str(obj)\n"
            ),
        },
        test_path="tests/test_serializer.py",
        test_code=(
            "from datetime import datetime\n"
            "from handlers.json_handler import CustomJsonSerializer\n\n"
            "def test_custom_serializer_datetime():\n"
            "    s = CustomJsonSerializer()\n"
            "    now = datetime(2026, 10, 18, 12, 0, 0)\n"
            "    res = s.serialize({'timestamp': now, 'status': 'ok'})\n"
            "    assert '2026-10-18T12:00:00' in res or '2026-10-18' in res\n"
        ),
        target_bug_file="core/base_handler.py",
        target_bug_symbol="class BaseSerializer",
    ))

    # -------------------------------------------------------------------------
    # Task 3: Cross-Module Config Validation Bug
    # Issue mentions auth token error, but bug is in configuration loader.
    # -------------------------------------------------------------------------
    tasks.append(BugTask(
        task_id="NC-TASK-003",
        title="Cross-Module JWT Configuration Validation",
        category="Module Import & Call",
        problem_statement=(
            "Configuration error: validate_jwt_settings in config/jwt_config.py fails to reject "
            "RS256 algorithm when public_key is empty or None. It currently only checks secret_key."
        ),
        files={
            "auth/token_manager.py": (
                "from config.jwt_config import validate_jwt_settings\n\n"
                "def create_auth_token(claims: dict, config: dict) -> str:\n"
                "    '''Generates verified auth token using configured algorithm.'''\n"
                "    if not validate_jwt_settings(config):\n"
                "        raise ValueError('Invalid JWT configuration')\n"
                "    return f'token.{claims.get(\"sub\")}'\n"
            ),
            "config/jwt_config.py": (
                "def validate_jwt_settings(config: dict) -> bool:\n"
                "    '''Validates token configuration parameters.'''\n"
                "    algo = config.get('algorithm', 'HS256')\n"
                "    secret = config.get('secret_key')\n"
                "    if not secret:\n"
                "        return False\n"
                "    return True\n"
            ),
            "auth/user_service.py": (
                "def get_user_claims(user_id: str) -> dict:\n"
                "    return {'sub': user_id, 'role': 'admin'}\n"
            ),
            "config/db_config.py": (
                "def get_database_url():\n"
                "    return 'sqlite:///:memory:'\n"
            ),
        },
        test_path="tests/test_jwt_config.py",
        test_code=(
            "import pytest\n"
            "from config.jwt_config import validate_jwt_settings\n"
            "from auth.token_manager import create_auth_token\n\n"
            "def test_rs256_requires_public_key():\n"
            "    bad_cfg = {'algorithm': 'RS256', 'secret_key': 'secret', 'public_key': None}\n"
            "    assert validate_jwt_settings(bad_cfg) is False\n\n"
            "def test_token_creation_fails_on_bad_rs256():\n"
            "    bad_cfg = {'algorithm': 'RS256', 'secret_key': 'secret', 'public_key': ''}\n"
            "    with pytest.raises(ValueError, match='Invalid JWT configuration'):\n"
            "        create_auth_token({'sub': 'user1'}, bad_cfg)\n"
        ),
        target_bug_file="config/jwt_config.py",
        target_bug_symbol="def validate_jwt_settings",
    ))

    # -------------------------------------------------------------------------
    # Task 4: Multi-hop Payment Charge Amount Boundary Check
    # Entrypoint in checkout route calls payment client where negative amount is permitted.
    # -------------------------------------------------------------------------
    tasks.append(BugTask(
        task_id="NC-TASK-004",
        title="Nested Multi-hop Payment Negative Charge Guard",
        category="Nested Call Chain",
        problem_statement=(
            "Bug: checkout route allows negative charge amounts to reach payment gateway. "
            "create_charge in payments/gateway_client.py must raise ValueError if amount <= 0."
        ),
        files={
            "web/checkout_route.py": (
                "from services.billing import process_order_payment\n\n"
                "def checkout_endpoint(order_id: str, amount: float) -> dict:\n"
                "    '''Web handler for processing checkout orders.'''\n"
                "    charge_id = process_order_payment(order_id, amount)\n"
                "    return {'order_id': order_id, 'charge_id': charge_id, 'status': 'paid'}\n"
            ),
            "services/billing.py": (
                "from payments.gateway_client import create_charge\n\n"
                "def process_order_payment(order_id: str, amount: float) -> str:\n"
                "    '''Orchestrates billing transaction for an order.'''\n"
                "    return create_charge(f'order_{order_id}', amount)\n"
            ),
            "payments/gateway_client.py": (
                "def create_charge(customer_ref: str, amount: float) -> str:\n"
                "    '''Submits charge to payment backend.'''\n"
                "    return f'ch_{customer_ref}_{int(amount)}'\n"
            ),
            "web/distractor_payment.py": (
                "def log_payment_amount(order_id, amount):\n"
                "    '''Logs payment checkout amount.'''\n"
                "    print(f'Order {order_id} checkout amount: {amount}')\n"
            ),
        },
        test_path="tests/test_payment.py",
        test_code=(
            "import pytest\n"
            "from payments.gateway_client import create_charge\n"
            "from web.checkout_route import checkout_endpoint\n\n"
            "def test_negative_charge_rejected():\n"
            "    with pytest.raises(ValueError, match='amount'):\n"
            "        create_charge('cust_1', -50.0)\n\n"
            "def test_zero_charge_rejected():\n"
            "    with pytest.raises(ValueError, match='amount'):\n"
            "        create_charge('cust_1', 0.0)\n\n"
            "def test_valid_charge_succeeds():\n"
            "    res = create_charge('cust_1', 100.0)\n"
            "    assert res.startswith('ch_')\n"
        ),
        target_bug_file="payments/gateway_client.py",
        target_bug_symbol="def create_charge",
    ))

    # -------------------------------------------------------------------------
    # Task 5: Database Query Boundary Slicing
    # sanitize_limit allows negative offset/limit causing index crashes.
    # -------------------------------------------------------------------------
    tasks.append(BugTask(
        task_id="NC-TASK-005",
        title="Query Builder Sanitization Boundary Check",
        category="Database Query Slicing",
        problem_statement=(
            "Issue: sanitize_limit in db/query_builder.py allows negative integers, "
            "causing SQL syntax errors. It must enforce limit >= 1 and clamp values greater than 100 to 100."
        ),
        files={
            "db/repository.py": (
                "from db.query_builder import sanitize_limit\n\n"
                "class UserRepository:\n"
                "    def fetch_users(self, limit: int = 10) -> str:\n"
                "        valid_limit = sanitize_limit(limit)\n"
                "        return f'SELECT * FROM users LIMIT {valid_limit}'\n"
            ),
            "db/query_builder.py": (
                "def sanitize_limit(limit: int) -> int:\n"
                "    '''Sanitizes user pagination limit.'''\n"
                "    return limit\n"
            ),
            "db/connection.py": (
                "def get_pool():\n"
                "    return {}\n"
            ),
            "api/user_controller.py": (
                "def get_user_page(page: int, limit: int):\n"
                "    return {'page': page, 'limit': limit}\n"
            ),
        },
        test_path="tests/test_query.py",
        test_code=(
            "import pytest\n"
            "from db.query_builder import sanitize_limit\n"
            "from db.repository import UserRepository\n\n"
            "def test_sanitize_limit_clamping():\n"
            "    assert sanitize_limit(150) == 100\n"
            "    assert sanitize_limit(50) == 50\n\n"
            "def test_sanitize_limit_negative():\n"
            "    with pytest.raises(ValueError, match='limit'):\n"
            "        sanitize_limit(-5)\n"
        ),
        target_bug_file="db/query_builder.py",
        target_bug_symbol="def sanitize_limit",
    ))

    return tasks


class EndToEndBenchmarkRunner:
    """
    Executes real multi-condition evaluation on physical hardware.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct", use_gpu: bool = True):
        self.model_id = model_id
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if self.model is None:
            print(f"[BENCHMARK] Loading {self.model_id} on {self.device}...")
            t0 = time.perf_counter()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map=self.device
            )
            print(f"[BENCHMARK] Model loaded in {time.perf_counter() - t0:.2f}s")

    def _setup_workspace(self, task: BugTask, target_dir: Path):
        for rel_path, content in task.files.items():
            file_path = target_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        test_file = target_dir / task.test_path
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(task.test_code, encoding="utf-8")

    def _run_pytest(self, workspace_dir: Path, test_rel_path: str) -> bool:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(workspace_dir)
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest", "-p", "no:launch_testing", test_rel_path],
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
                env=env,
                timeout=15
            )
            return res.returncode == 0
        except Exception:
            return False

    def build_prompt_raw(self, workspace_dir: Path, task: BugTask) -> Tuple[str, float]:
        t0 = time.perf_counter()
        full_text = []
        for rel_path in sorted(task.files.keys()):
            content = (workspace_dir / rel_path).read_text(encoding="utf-8")
            full_text.append(f"FILE: {rel_path}\n```python\n{content}\n```")
        context_str = "\n\n".join(full_text)
        prep_ms = (time.perf_counter() - t0) * 1000

        prompt = (
            f"You are an expert autonomous software engineer.\n"
            f"Repository Files:\n{context_str}\n\n"
            f"Issue to fix: {task.problem_statement}\n"
            f"Provide the fix using exact SEARCH/REPLACE format:\n"
            f"FILE: <file_path>\n<<<<<<< SEARCH\n=======\n>>>>>>>\n"
        )
        return prompt, prep_ms

    def build_prompt_bm25(self, workspace_dir: Path, task: BugTask) -> Tuple[str, float]:
        t0 = time.perf_counter()
        q_tokens = set(task.problem_statement.lower().replace(".", " ").replace("/", " ").split())
        scores = {}
        for rel_path in task.files.keys():
            content = (workspace_dir / rel_path).read_text(encoding="utf-8")
            words = content.lower().split()
            score = sum(words.count(q) for q in q_tokens)
            scores[rel_path] = score

        top_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        context_parts = []
        for rel_path, _ in top_files:
            content = (workspace_dir / rel_path).read_text(encoding="utf-8")
            context_parts.append(f"FILE: {rel_path}\n```python\n{content}\n```")
        context_str = "\n\n".join(context_parts)
        prep_ms = (time.perf_counter() - t0) * 1000

        prompt = (
            f"You are an expert autonomous software engineer.\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"Issue to fix: {task.problem_statement}\n"
            f"Provide the fix using exact SEARCH/REPLACE format:\n"
            f"FILE: <file_path>\n<<<<<<< SEARCH\n=======\n>>>>>>>\n"
        )
        return prompt, prep_ms

    def build_prompt_graph_unconstrained(self, workspace_dir: Path, task: BugTask) -> Tuple[str, float]:
        t0 = time.perf_counter()
        builder = RepoGraphBuilder(str(workspace_dir))
        graph = builder.build()
        slicer = CodeGraphSlicer(graph)
        seeds = slicer.find_seed_nodes(task.problem_statement)
        # Unconstrained 1-hop traversal without knapsack packing
        subgraph = slicer.extract_subgraph(seeds, k_hops=1)
        
        # Serialize entire subgraph without token budgeting
        lines = []
        for node in subgraph.nodes:
            data = subgraph.nodes[node]
            lines.append(f"Node: {node} ({data.get('type')})\n{data.get('source', '')}\n")
        context_str = "\n".join(lines)
        prep_ms = (time.perf_counter() - t0) * 1000

        prompt = (
            f"You are an expert autonomous software engineer.\n"
            f"Graph Context:\n{context_str}\n\n"
            f"Issue to fix: {task.problem_statement}\n"
            f"Provide the fix using exact SEARCH/REPLACE format:\n"
            f"FILE: <file_path>\n<<<<<<< SEARCH\n=======\n>>>>>>>\n"
        )
        return prompt, prep_ms

    def build_prompt_neurocode(self, workspace_dir: Path, task: BugTask) -> Tuple[str, float]:
        t0 = time.perf_counter()
        builder = RepoGraphBuilder(str(workspace_dir))
        graph = builder.build()
        slicer = CodeGraphSlicer(graph)
        seeds = slicer.find_seed_nodes(task.problem_statement)
        subgraph = slicer.extract_subgraph(seeds, k_hops=1)
        slice_text = slicer.format_slice_for_llm(subgraph, seeds, query=task.problem_statement, max_tokens=1500)
        prep_ms = (time.perf_counter() - t0) * 1000

        prompt = (
            f"You are an expert autonomous software engineer.\n"
            f"{slice_text}\n\n"
            f"Issue to fix: {task.problem_statement}\n"
            f"Provide the fix using exact SEARCH/REPLACE format:\n"
            f"FILE: <file_path>\n<<<<<<< SEARCH\n=======\n>>>>>>>\n"
        )
        return prompt, prep_ms

    def evaluate_task_method(self, task: BugTask, method: str) -> MethodEvaluationResult:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            self._setup_workspace(task, workspace)

            # 1. Verify that test originally FAILS
            initial_test_passed = self._run_pytest(workspace, task.test_path)
            assert not initial_test_passed, f"Task {task.task_id} test passed before bug patch! Invalid test."

            builder = RepoGraphBuilder(str(workspace))
            graph = builder.build()

            # 2. Build Prompt according to Method
            if method == "Raw Context Dump":
                prompt, prep_ms = self.build_prompt_raw(workspace, task)
            elif method == "BM25 Lexical RAG":
                prompt, prep_ms = self.build_prompt_bm25(workspace, task)
            elif method == "Graph Unconstrained":
                prompt, prep_ms = self.build_prompt_graph_unconstrained(workspace, task)
            elif method == "NeuroCode (Our Method)":
                prompt, prep_ms = self.build_prompt_neurocode(workspace, task)
            else:
                raise ValueError(f"Unknown method {method}")

            captured_target = task.target_bug_symbol in prompt and task.target_bug_file in prompt

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
            input_tokens = inputs.input_ids.shape[1]

            # 3. Model Inference on Device
            if self.use_gpu:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            t0_infer = time.perf_counter()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=220,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            infer_ms = (time.perf_counter() - t0_infer) * 1000
            total_ms = prep_ms + infer_ms
            output_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            peak_vram_mb = 0.0
            if self.use_gpu:
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            # 4. Patch Application & Verification (with AST fallback)
            patch_res = PatchApplier.extract_patch(response_text, graph=graph, repo_dir=str(workspace))
            valid_patch = patch_res.success
            patch_applied = False
            test_passed = False
            err_msg = None

            if valid_patch:
                target_file_path = workspace / patch_res.file_path
                if target_file_path.exists():
                    applied, msg = PatchApplier.apply_patch_to_file(
                        str(target_file_path),
                        patch_res.search_block,
                        patch_res.replace_block
                    )
                    patch_applied = applied
                    if applied:
                        test_passed = self._run_pytest(workspace, task.test_path)
                    else:
                        err_msg = msg
                else:
                    err_msg = f"Target file does not exist: {patch_res.file_path}"
            else:
                err_msg = patch_res.error_message

            return MethodEvaluationResult(
                method=method,
                task_id=task.task_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prep_time_ms=prep_ms,
                infer_time_ms=infer_ms,
                total_time_ms=total_ms,
                peak_vram_mb=peak_vram_mb,
                captured_target=captured_target,
                valid_patch_format=valid_patch,
                patch_applied=patch_applied,
                test_passed=test_passed,
                model_output=response_text,
                error_message=err_msg
            )


def run_full_end_to_end_benchmark() -> List[MethodEvaluationResult]:
    print("==========================================================================")
    print("       NEUROCODE VERIFIED END-TO-END BUG RESOLUTION BENCHMARK            ")
    print("==========================================================================\n")

    tasks = get_curated_benchmark_tasks()
    print(f"Total Evaluated Tasks: {len(tasks)}")

    runner = EndToEndBenchmarkRunner()
    runner.load_model()

    methods = [
        "Raw Context Dump",
        "BM25 Lexical RAG",
        "Graph Unconstrained",
        "NeuroCode (Our Method)"
    ]

    all_results = []

    for task in tasks:
        print(f"\n--------------------------------------------------------------------------")
        print(f"Executing Task: [{task.task_id}] {task.title} ({task.category})")
        print(f"--------------------------------------------------------------------------")
        for method in methods:
            print(f"  -> Running Condition: {method:25s} ... ", end="", flush=True)
            res = runner.evaluate_task_method(task, method)
            all_results.append(res)
            status = "RESOLVED [PASS]" if res.test_passed else ("APPLIED [FAIL TEST]" if res.patch_applied else "NO PATCH")
            print(f"{status:18s} | In Tokens: {res.input_tokens:4d} | Target In Context: {res.captured_target} | VRAM: {res.peak_vram_mb:.1f}MB")

    return all_results


if __name__ == "__main__":
    results = run_full_end_to_end_benchmark()
