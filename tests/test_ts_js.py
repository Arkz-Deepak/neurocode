"""Tests for TypeScript and JavaScript AST parsing in NeuroCode."""
import pytest
from src.graph.parser import parse_js_ts_file, parse_file
from src.graph.builder import RepoGraphBuilder


def test_parse_typescript_classes_and_methods(tmp_path):
    ts_code = """
import { Injectable } from '@nestjs/common';
import { Logger } from './logger';

export class OrderService extends BaseService {
    private logger: Logger;

    async createOrder(userId: string, amount: number): Promise<boolean> {
        this.logger.log('Creating order');
        return true;
    }
}

export const formatPrice = (price: number) => {
    return '$' + price.toFixed(2);
};
"""
    ts_file = tmp_path / "order.service.ts"
    ts_file.write_text(ts_code, encoding="utf-8")

    parsed = parse_file(str(ts_file))
    assert parsed.file_path == str(ts_file)
    assert len(parsed.imports) >= 1
    assert len(parsed.classes) == 1
    assert parsed.classes[0].name == "OrderService"
    assert "BaseService" in parsed.classes[0].bases
    assert len(parsed.classes[0].methods) == 1
    assert parsed.classes[0].methods[0].name == "createOrder"

    # Arrow function
    assert any(f.name == "formatPrice" for f in parsed.functions)


def test_parse_javascript_functions_and_calls(tmp_path):
    js_code = """
import React, { useState } from 'react';

function Counter() {
    const [count, setCount] = useState(0);
    return count;
}

export default Counter;
"""
    js_file = tmp_path / "Counter.jsx"
    js_file.write_text(js_code, encoding="utf-8")

    parsed = parse_file(str(js_file))
    assert len(parsed.functions) == 1
    assert parsed.functions[0].name == "Counter"
    assert "useState" in parsed.functions[0].calls
