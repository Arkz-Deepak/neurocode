import time
import json
import threading
import urllib.request
import pytest
from src.proxy import ThreadedHTTPServer, NeuroCodeProxyHandler

@pytest.fixture(scope="module")
def proxy_server():
    server_address = ("127.0.0.1", 8999)
    NeuroCodeProxyHandler.upstream_url = "http://localhost:11434/v1"
    NeuroCodeProxyHandler.token_budget = 1200
    httpd = ThreadedHTTPServer(server_address, NeuroCodeProxyHandler)
    
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    yield "http://127.0.0.1:8999"
    httpd.shutdown()

def test_proxy_health(proxy_server):
    req = urllib.request.Request(f"{proxy_server}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["status"] == "healthy"
        assert "NeuroCode" in data["engine"]

def test_proxy_models(proxy_server):
    req = urllib.request.Request(f"{proxy_server}/v1/models")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert "data" in data
        assert any(m["id"] == "neurocode-auto" for m in data["data"])

def test_proxy_slicing_chat_completion(proxy_server):
    # Construct a large code payload (> 4,000 characters)
    large_code = "\n".join([f"def function_{i}(x):\n    return x * {i}\n" for i in range(120)])
    assert len(large_code) > 4000

    payload = {
        "model": "neurocode-auto",
        "messages": [
            {"role": "system", "content": "You are a code repair assistant."},
            {"role": "user", "content": f"Please fix the bug in function_42:\n```python\n{large_code}\n```"}
        ]
    }
    data_bytes = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{proxy_server}/v1/chat/completions",
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode())
        assert "choices" in res
        content = res["choices"][0]["message"]["content"]
        assert "NeuroCode Proxy Active" in content
        assert "Bounded Context" in content
