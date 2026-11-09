"""Tests for AST Subtree Grafting (neurocode_graft)."""
import pytest
from src.agent.graft import graft_symbol


def test_graft_python_function(tmp_path):
    py_code = """
class BillingService:
    def calculate_discount(self, total: float) -> float:
        return total * 0.05

    def get_tier(self) -> str:
        return "STANDARD"
"""
    py_file = tmp_path / "billing.py"
    py_file.write_text(py_code, encoding="utf-8")

    replacement = """def calculate_discount(self, total: float) -> float:
    # upgraded VIP discount logic
    if total > 1000:
        return total * 0.20
    return total * 0.10"""

    res = graft_symbol(
        repo_path=str(tmp_path),
        file_path="billing.py",
        symbol="calculate_discount",
        replacement_code=replacement,
        apply=True
    )

    assert res["success"] is True
    assert res["syntax_valid"] is True
    assert res["applied"] is True
    assert "upgraded VIP discount" in res["diff"]

    # Verify file content on disk
    updated_content = py_file.read_text(encoding="utf-8")
    assert "upgraded VIP discount logic" in updated_content
    assert "STANDARD" in updated_content


def test_graft_syntax_error_rejection(tmp_path):
    py_code = """
def simple_add(a, b):
    return a + b
"""
    py_file = tmp_path / "math_ops.py"
    py_file.write_text(py_code, encoding="utf-8")

    # Deliberate syntax error: missing closing parenthesis and invalid statement
    bad_replacement = """def simple_add(a, b:
    return a +++ %%% b"""

    res = graft_symbol(
        repo_path=str(tmp_path),
        file_path="math_ops.py",
        symbol="simple_add",
        replacement_code=bad_replacement,
        apply=True
    )

    assert res["success"] is False
    assert res["syntax_valid"] is False

    # Original file must remain untouched
    content = py_file.read_text(encoding="utf-8")
    assert "return a + b" in content


def test_graft_typescript_method(tmp_path):
    ts_code = """
export class MathHelper {
    public multiply(x: number, y: number): number {
        return x * y;
    }
}
"""
    ts_file = tmp_path / "math.ts"
    ts_file.write_text(ts_code, encoding="utf-8")

    replacement = """public multiply(x: number, y: number): number {
    console.log('multiplying', x, y);
    return x * y;
}"""

    res = graft_symbol(
        repo_path=str(tmp_path),
        file_path="math.ts",
        symbol="MathHelper.multiply",
        replacement_code=replacement,
        apply=True
    )

    assert res["success"] is True
    assert res["syntax_valid"] is True
    updated = ts_file.read_text(encoding="utf-8")
    assert "console.log('multiplying', x, y);" in updated
