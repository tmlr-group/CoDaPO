---
sidebar_label: "Adding a New Tool"
sidebar_position: 2
---

# Adding a New Tool

This guide walks you through extending AlphaApollo with a custom tool. By the end you will have a fully integrated tool that the LLM can invoke during multi-turn reasoning.

:::info Prerequisite reading

- [tools.md](../core-modules/tools.md) for an overview of the existing tool framework
- [agent-system.md](../core-modules/agent-system.md) for how tools fit into the environment loop
  :::

## Architecture at a Glance

```text
@tool decorator (core.py)
    │
    ▼
ToolGroup (core.py)          ← auto-discovers decorated methods
    │
    ▼
InformalMathToolGroup         ← concrete group shipped with AlphaApollo
(manager.py)                     (python_code, local_rag)
    │
    ▼
BaseTextEnv._execute_tool()  ← environment dispatches calls at runtime
```

Key design rules:

| Rule                                                                         | Reason                                                                                                                                                                                                        |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Every `@tool` method must return `{"text_result": str, "score": int\|float}` | The environment wraps `text_result` in `<tool_response>` tags and feeds it back to the model. `score` is used for intermediate diagnostics.                                                                   |
| The **function name** must match the **XML tag name** used by the model      | The projection layer parses `<tool_name>…</tool_name>` tags, and the environment uses the same name to dispatch. See `core/tools/manager.py` — _"you MUST align the tool calling tokens with the tool name"_. |
| Tools are stateless across episodes                                          | Ground-truth or episode-specific context is injected via `set_ground_truth()` at each `reset()`.                                                                                                              |

## Step-by-Step Guide

### Step 1 — Implement the Tool Logic

Create a new file under `alphaapollo/core/tools/`. The file should contain a pure function (or class) that performs the actual work and returns a result dictionary.

```python
# alphaapollo/core/tools/my_tool.py

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def execute_my_tool(query: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Core logic for the custom tool.

    Returns:
        Dict with at least:
        - "result": str   — human-readable result
        - "status": str   — "success" | "error"
    """
    try:
        # ... your domain-specific logic here ...
        output = f"Processed: {query}"
        return {"result": output, "status": "success"}
    except Exception as e:
        logger.error(f"my_tool failed: {e}")
        return {"result": str(e), "status": "error"}
```

:::tip
Keep this file free of any AlphaApollo framework imports — it makes unit-testing much easier.
:::

### Step 2 — Register in a ToolGroup

You have two options:

**Option A — Add to the existing `InformalMathToolGroup`** (quickest path):

```python
# In alphaapollo/core/tools/manager.py, inside class InformalMathToolGroup:

from alphaapollo.core.tools.my_tool import execute_my_tool

@tool
def my_tool(self, query: str) -> Dict[str, Any]:
    """Invoke the custom tool."""
    if not query or not query.strip():
        return {"text_result": "No query provided.", "score": 0}

    result = execute_my_tool(query=query, timeout=30)
    status = result.get("status", "error")
    return {
        "text_result": json.dumps(result),
        "score": 1 if status == "success" else 0,
    }
```

**Option B — Create a new `ToolGroup` subclass** (recommended for a self-contained domain):

```python
# alphaapollo/core/tools/my_tool_group.py

import json
from typing import Dict, Any
from alphaapollo.core.tools.core import ToolGroup, tool
from alphaapollo.core.tools.my_tool import execute_my_tool

class MyToolGroup(ToolGroup):
    """A custom tool group."""

    def __init__(self, tool_config: dict = None):
        tool_config = tool_config or {}
        self.timeout = tool_config.get("my_tool_timeout", 10)
        super().__init__(name="MyToolGroup")

    @tool
    def my_tool(self, query: str) -> Dict[str, Any]:
        result = execute_my_tool(query=query, timeout=self.timeout)
        return {
            "text_result": json.dumps(result),
            "score": 1 if result["status"] == "success" else 0,
        }
```

If you chose Option B, register the new group in the environment's `__init__`:

```python
# In the domain environment (e.g. env.py)
self.my_tool_group = MyToolGroup(tool_config=tool_config)
self.init_tool_groups([self.tool_group, self.my_tool_group])
```

:::warning
The `@tool` decorator uses `func.__name__` as the tool name. The method **must** be named exactly the same as the XML tag the model will emit (e.g. `my_tool` ↔ `<my_tool>…</my_tool>`).
:::

### Step 3 — Register the Tool Pattern in the Environment
:::warning Update both environments
AlphaApollo has **two parallel environment packages**: `informal_math_training/` and `informal_math_evolving/`. You must add your tool pattern and dispatch logic to **both** `env.py` files, and add the tool token to **both** `projection.py` files. Missing either will cause the tool to be unavailable in that workflow.
:::
Open the environment's `env.py` (e.g. `core/environments/informal_math_training/env.py`) and add an entry to `TOOL_PATTERNS`:

```python
TOOL_PATTERNS = [
    ("python_code",          r"<python_code>(.*?)</python_code>"),
    ("local_rag",            r"<local_rag>(.*?)</local_rag>"),
    # ↓ your new tool
    ("my_tool",              r"<my_tool>(.*?)</my_tool>"),
]
```

Then add the dispatch branch inside the `step()` method's tool-call loop:

```python
elif tool_name == "my_tool":
    observation = self._execute_tool(
        "InformalMathToolGroup",   # or "MyToolGroup" if Option B
        "my_tool",
        {"query": tool_input},
    )
    tool_info = {
        "tool_calling": True,
        "tool_group": "InformalMathToolGroup",
        "tool_name": "my_tool",
        "tool_input": tool_input,
        "data_source": self.data_source,
    }
```
:::info Argument convention
`_execute_tool()` passes the third argument to `ToolGroup.execute_tool()`. When a single `dict` is passed, `execute_tool()` automatically unpacks it as `**kwargs` to the `@tool` method. This means the dict keys must match the method’s parameter names.
:::
### Step 4 — Update the Projection Layer

Open `core/environments/informal_math_training/projection.py` and add the new tag to `TOOL_CALLING_TOKENS`:

```python
TOOL_CALLING_TOKENS = [
    "python_code",
    "local_rag",
    "my_tool",              # ← add here
]
```

This list is used by the projection function to:

1. Trim the LLM output at the first closing tag (to prevent hallucinated multi-tool outputs).
2. Validate that the action contains at most one tool call tag or one `<answer>` tag (not both).

No further changes to `informal_math_training_projection()` are needed — the logic is driven entirely by `TOOL_CALLING_TOKENS`.

### Step 5 — Update Prompt Templates

Add instructions for the new tool to the prompt templates in `alphaapollo/core/environments/prompts/`. The prompt tells the model the tool exists and how to call it:

```text
## my_tool
Use <my_tool>your query here</my_tool> to invoke the custom tool.
The tool returns a JSON object with a "result" field.
```

If the tool should only be available when a config flag is set, gate it in the prompt-building function:

```python
if tool_config.get("enable_my_tool", False):
    prompt += MY_TOOL_INSTRUCTIONS
```

### Step 6 — Add Configuration Support

Add a flag to your YAML config so the tool can be enabled or disabled at runtime:

```yaml
# In an RL/SFT/Test config YAML
- env.informal_math.enable_my_tool=true
- env.informal_math.my_tool_timeout=15
```

Read the flag in your `ToolGroup.__init__`:

```python
self.enable_my_tool = tool_config.get("enable_my_tool", False)
```

And guard the `@tool` method body:

```python
@tool
def my_tool(self, query: str) -> Dict[str, Any]:
    if not self.enable_my_tool:
        return {"text_result": "Tool not enabled.", "score": 0}
    ...
```

## Testing Your Tool

1. **Unit test** — call the underlying function directly:

   ```python
   # tests/test_my_tool.py
   import pytest
   from alphaapollo.core.tools.my_tool import execute_my_tool

   def test_my_tool_success():
       result = execute_my_tool("test query")
       assert result["status"] == "success"
       assert "result" in result

   def test_my_tool_empty_input():
       result = execute_my_tool("")
       assert result["status"] == "error"
   ```

2. **Integration test** — instantiate the `ToolGroup` and dispatch:

   ```python
   # tests/test_my_tool_integration.py
   import pytest
   from alphaapollo.core.tools.manager import InformalMathToolGroup

   @pytest.fixture
   def tool_group():
       return InformalMathToolGroup(
           tool_config={"enable_my_tool": True, "enable_python_code": False}
       )

   def test_dispatch(tool_group):
       out = tool_group.execute_tool("my_tool", query="test")
       assert "text_result" in out
       assert out["score"] in (0, 1)

   def test_tool_names(tool_group):
       assert "my_tool" in tool_group.get_tool_names()
   ```

3. **End-to-end** — run the terminal demo with an updated config:

   ```bash
   python -m alphaapollo.workflows.test --config examples/configs/test_informal_math.yaml
   ```

## Checklist

| #   | Item                                | Where                                                    |
| --- | ----------------------------------- | -------------------------------------------------------- |
| 1   | Implement tool logic                | `core/tools/my_tool.py`                                  |
| 2   | Add `@tool` method to a `ToolGroup` | `core/tools/manager.py` or new file                      |
| 3   | Add `TOOL_PATTERNS` entry           | `core/environments/*/env.py` (**both** training & evolving) |
| 4   | Add to `TOOL_CALLING_TOKENS`        | `core/environments/*/projection.py` (**both** packages)  |
| 5   | Update prompt templates             | `core/environments/prompts/`                             |
| 6   | Add YAML config flags               | `examples/configs/*.yaml`                                |
| 7   | Write unit + integration tests      | `tests/`                                                 |
