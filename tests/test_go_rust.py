"""Tests for Go and Rust multi-language AST parsing and grafting."""
import pytest
from src.graph.parser import parse_go_file, parse_rust_file, parse_file
from src.agent.graft import graft_symbol


def test_parse_go_file(tmp_path):
    go_code = """package main
import "fmt"

type Service struct {
    host string
}

func (s *Service) Start() {
    fmt.Println("Starting service")
}

func CalculateTax(amount float64) float64 {
    return amount * 0.15
}
"""
    go_file = tmp_path / "main.go"
    go_file.write_text(go_code, encoding="utf-8")

    parsed = parse_file(str(go_file))
    assert len(parsed.imports) == 1
    assert parsed.imports[0].module == "fmt"
    assert len(parsed.classes) == 1
    assert parsed.classes[0].name == "Service"
    assert any(f.name == "Start" for f in parsed.functions)
    assert any(f.name == "CalculateTax" for f in parsed.functions)


def test_parse_rust_file(tmp_path):
    rs_code = """use std::io;

pub struct Wallet {
    pub balance: u64,
}

impl Wallet {
    pub fn deposit(&mut self, amt: u64) {
        self.balance += amt;
    }
}

pub fn get_fee() -> u64 {
    5
}
"""
    rs_file = tmp_path / "lib.rs"
    rs_file.write_text(rs_code, encoding="utf-8")

    parsed = parse_file(str(rs_file))
    assert len(parsed.imports) == 1
    assert len(parsed.classes) == 1
    assert parsed.classes[0].name == "Wallet"
    assert any(f.name == "deposit" for f in parsed.functions)
    assert any(f.name == "get_fee" for f in parsed.functions)


def test_graft_go_function(tmp_path):
    go_code = """package main

func CalculateTax(amount float64) float64 {
    return amount * 0.10
}
"""
    go_file = tmp_path / "tax.go"
    go_file.write_text(go_code, encoding="utf-8")

    rep = """func CalculateTax(amount float64) float64 {
    // 20% rate
    return amount * 0.20
}"""
    res = graft_symbol(
        repo_path=str(tmp_path),
        file_path="tax.go",
        symbol="CalculateTax",
        replacement_code=rep,
        apply=True
    )
    assert res["success"] is True
    assert res["syntax_valid"] is True
    assert "0.20" in go_file.read_text(encoding="utf-8")
