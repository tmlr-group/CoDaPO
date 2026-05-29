# Backend Logic Refinement Plan (`examples/demo`)

## 1. Objective
Refine backend behavior in `examples/demo/web_informal_math_training.py` for correctness, robustness, and predictable multi-turn streaming without changing frontend presentation.

## 2. Target Outcomes
1. Stable SSE event contract for `/api/chat/stream` and `/api/solve/stream`.
2. Reliable conversation lifecycle management (`conversation_id`, reset semantics, bounded memory).
3. Better error handling and observability for model, environment, and stream failures.
4. Deterministic extraction of assistant-visible answer content from model actions.
5. No regressions to existing API behavior used by the current web client.

## 3. Scope
- In scope:
  - `LLMClient` request/streaming behavior
  - `DemoRuntime` state handling (`solve`, `solve_stream`, `chat_stream`)
  - SSE and JSON API behavior in `DemoHandler`
  - configuration guards in `merge_config`
- Out of scope:
  - UI/HTML/CSS/JS changes (`examples/demo/web_ui/index.html`)
  - environment internals under `alphaapollo/core/...`
  - server framework replacement

## 4. Implementation Plan

### Milestone A: API and Event Contract Hardening
Files:
- `examples/demo/web_informal_math_training.py`

Tasks:
1. Define and enforce required fields per SSE event type (`start`, `step_start`, `llm_token`, `llm_end`, `step_end`, `done`, `error`).
2. Ensure JSON serialization is resilient for all event payloads.
3. Keep `/api/chat/stream` and `/api/solve/stream` contracts explicit and backward compatible.

Acceptance criteria:
- Stream consumers receive consistent event shapes across runs.
- No unhandled serialization exceptions in stream loop.

### Milestone B: Conversation State and Memory Controls
Files:
- `examples/demo/web_informal_math_training.py`

Tasks:
1. Add retention policy for `self.conversations` (max conversations, max turns per conversation).
2. Keep current `conversation_id` behavior while preventing unbounded memory growth.
3. Clarify reset semantics when `reset=True` vs missing/unknown `conversation_id`.

Acceptance criteria:
- Follow-up turns remain correct.
- Process memory growth is bounded during prolonged usage.

### Milestone C: LLM Streaming Robustness
Files:
- `examples/demo/web_informal_math_training.py`

Tasks:
1. Strengthen `LLMClient.generate_stream` against partial/empty deltas and transient API anomalies.
2. Add guarded handling for upstream timeout/retry errors.
3. Ensure stream completion always emits terminal event (`done` or `error`).

Acceptance criteria:
- Streams do not hang silently on upstream issues.
- Client always receives explicit terminal status.

### Milestone D: Action Parsing and Answer Extraction
Files:
- `examples/demo/web_informal_math_training.py`

Tasks:
1. Harden `_extract_action_blocks` for malformed tag structures.
2. Improve assistant reply extraction priority in `chat_stream`:
   - prefer `<answer>` blocks
   - fallback safely to raw action text when needed
3. Normalize tool response extraction behavior in `_extract_tool_response`.

Acceptance criteria:
- Assistant-visible reply is deterministic for same model output.
- Malformed model tags do not crash a step.

### Milestone E: Error Handling and Logging
Files:
- `examples/demo/web_informal_math_training.py`

Tasks:
1. Add lightweight structured logs for request start/end, conversation id, step count, and terminal status.
2. Separate client errors (4xx) from server/runtime errors (5xx) with clearer payloads.
3. Preserve graceful handling for disconnects (`BrokenPipeError`, `ConnectionResetError`).

Acceptance criteria:
- Failure modes are diagnosable from logs.
- API responses are consistent with HTTP status class.

## 5. Validation Checklist
1. Run server:
   - `MODE=web bash examples/demo/run_terminal_demo_vllm.sh`
2. API checks:
   - `POST /api/chat/stream` with new message
   - follow-up with same `conversation_id`
   - `reset=true` behavior check
   - invalid payload tests (`missing message`, malformed JSON)
3. Failure checks:
   - simulated LLM endpoint failure/timeouts
   - client disconnect mid-stream
4. Verify terminal event always delivered (`done` or `error`).

## 6. Risks and Mitigations
1. Contract changes may break current frontend:
   - mitigation: keep event names and core fields backward compatible.
2. Additional guards may mask real bugs:
   - mitigation: include explicit error payloads and logs instead of silent fallback.
3. Memory retention policy may remove useful context:
   - mitigation: use conservative limits and document defaults.

## 7. Definition of Done
1. All backend milestones A-E meet acceptance criteria.
2. Multi-turn chat and solve streaming remain functional with existing client.
3. Runtime handles malformed inputs and upstream failures without crashing.
