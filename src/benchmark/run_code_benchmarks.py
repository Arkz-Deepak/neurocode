#!/usr/bin/env python3
"""
Autonomous Multi-Benchmark Runner for HumanEval and MBPP.
Executes code generation and runs automated Python unit test verification (Pass@1).
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_humaneval_evaluation(model, tokenizer, data_path: Path, output_path: Path, max_tasks: int = 164) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"      RUNNING BENCHMARK: HumanEval (N = {max_tasks} Tasks)")
    print("=" * 70)

    with open(data_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()][:max_tasks]

    results = []
    passed_count = 0
    t0_suite = time.perf_counter()

    for i, task in enumerate(tasks):
        tid = task["task_id"]
        prompt = task["prompt"]
        entry_point = task["entry_point"]
        test_code = task["test"]

        messages = [
            {"role": "system", "content": "You are an expert Python engineer. Complete the function accurately. Provide only valid Python code matching the prompt."},
            {"role": "user", "content": f"Complete the following Python function:\n\n{prompt}"}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        t0_gen = time.perf_counter()
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_ms = (time.perf_counter() - t0_gen) * 1000
        gen_code = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Clean code
        clean_code = gen_code
        if "```python" in gen_code:
            clean_code = gen_code.split("```python")[1].split("```")[0]
        elif "```" in gen_code:
            clean_code = gen_code.split("```")[1].split("```")[0]

        full_prog = prompt + "\n" + clean_code + "\n" + test_code + f"\ncheck({entry_point})\n"

        test_passed = False
        error_msg = ""
        try:
            exec_globals = {}
            exec(full_prog, exec_globals)
            test_passed = True
            passed_count += 1
        except Exception as e:
            test_passed = False
            error_msg = str(e)

        row = {
            "task_id": tid,
            "test_passed": test_passed,
            "gen_ms": round(gen_ms, 2),
            "tokens": len(clean_code.split()),
            "error": error_msg
        }
        results.append(row)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        stat = "PASS [OK]" if test_passed else "FAIL"
        if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
            print(f"[{i+1:3d}/{len(tasks)}] {tid:20s} | {stat:9s} | Running Pass@1: {passed_count}/{i+1} ({passed_count/(i+1)*100:.1f}%)")

    total_time = (time.perf_counter() - t0_suite) / 60
    pass_rate = (passed_count / len(tasks)) * 100
    print("=" * 70)
    print(f"HumanEval Evaluation Complete in {total_time:.1f}m: {passed_count}/{len(tasks)} Passed ({pass_rate:.1f}%)")
    print("=" * 70)
    return {"tasks": len(tasks), "passed": passed_count, "pass_rate": pass_rate}


def run_mbpp_evaluation(model, tokenizer, data_path: Path, output_path: Path, max_tasks: int = 200) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"      RUNNING BENCHMARK: MBPP (N = {max_tasks} Tasks)")
    print("=" * 70)

    with open(data_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if "test_list" in json.loads(line)][:max_tasks]

    results = []
    passed_count = 0
    t0_suite = time.perf_counter()

    for i, task in enumerate(tasks):
        tid = task["task_id"]
        text_desc = task["text"]
        test_list = task["test_list"]

        prompt = f"Write a Python function to solve this problem:\n{text_desc}\n\nYour code should pass the following assertions:\n" + "\n".join(test_list[:2])
        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Write only valid, working Python code with no markdown formatting or text."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        t0_gen = time.perf_counter()
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_ms = (time.perf_counter() - t0_gen) * 1000
        gen_code = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        clean_code = gen_code
        if "```python" in gen_code:
            clean_code = gen_code.split("```python")[1].split("```")[0]
        elif "```" in gen_code:
            clean_code = gen_code.split("```")[1].split("```")[0]

        test_passed = False
        error_msg = ""
        try:
            exec_globals = {}
            exec(clean_code, exec_globals)
            for assertion in test_list:
                exec(assertion, exec_globals)
            test_passed = True
            passed_count += 1
        except Exception as e:
            test_passed = False
            error_msg = str(e)

        row = {
            "task_id": tid,
            "test_passed": test_passed,
            "gen_ms": round(gen_ms, 2),
            "tokens": len(clean_code.split()),
            "error": error_msg
        }
        results.append(row)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        stat = "PASS [OK]" if test_passed else "FAIL"
        if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
            print(f"[{i+1:3d}/{len(tasks)}] Task {tid:<5d} | {stat:9s} | Running Pass@1: {passed_count}/{i+1} ({passed_count/(i+1)*100:.1f}%)")

    total_time = (time.perf_counter() - t0_suite) / 60
    pass_rate = (passed_count / len(tasks)) * 100
    print("=" * 70)
    print(f"MBPP Evaluation Complete in {total_time:.1f}m: {passed_count}/{len(tasks)} Passed ({pass_rate:.1f}%)")
    print("=" * 70)
    return {"tasks": len(tasks), "passed": passed_count, "pass_rate": pass_rate}


if __name__ == "__main__":
    print("[INIT] Multi-Benchmark Harness Ready.")
