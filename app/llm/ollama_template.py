"""Prepare a local template alias for Ollama's ToolFunction rendering defect.

No weights, sampling settings, or tool definitions change. Preparation happens
before readiness, never on a conversation turn, and never pulls a model.
"""
from dataclasses import replace
import hashlib
import json

from app.llm.ollama_client import OllamaConfig


async def prepare_tool_template(config, client):
    if not isinstance(config, OllamaConfig):
        return config
    response = await client.post(f"{config.url}/api/show", json={"model": config.model})
    response.raise_for_status()
    source = response.json()
    template = source.get("template", "")
    # Tool.String has a value receiver; ToolFunction.String has a pointer receiver
    # in Ollama 0.14.2. Rendering the whole Tool preserves valid JSON in both cases.
    broken = '{"type": "function", "function": {{ .Function }}}'
    if broken not in template:
        return config
    fixed = template.replace(broken, '{{ . }}')
    # The stock example uses an unquoted placeholder for a JSON string. Small
    # models can reproduce it literally and Ollama then drops the malformed call.
    fixed = fixed.replace('"name": <function-name>', '"name": "<function-name>"')
    identity = json.dumps({key: source.get(key) for key in
                           ("modelfile", "parameters", "system")}, sort_keys=True) + fixed
    alias = "aria-tools-" + hashlib.sha256(identity.encode()).hexdigest()[:16] + ":latest"
    response = await client.post(f"{config.url}/api/show", json={"model": alias})
    if response.status_code == 404:
        response = await client.post(f"{config.url}/api/create", json={
            "model": alias, "from": config.model, "template": fixed, "stream": False})
        response.raise_for_status()
        if response.json().get("status") != "success":
            raise RuntimeError("Local Ollama template preparation did not complete")
    else:
        response.raise_for_status()
        if response.json().get("template") != fixed:
            raise RuntimeError("Prepared Ollama template does not match its source")
    return replace(config, model=alias)
