#!/usr/bin/env python3
"""Self-contained web UI for informal_math_training demo.

No dependency on Biomni-Web code. This serves a simple HTML interface and a JSON API.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphaapollo.core.environments.env_manager import InformalMathTrainingEnvironmentManager
from alphaapollo.core.environments.informal_math_training import (
    build_informal_math_training_envs,
    informal_math_training_projection,
)


HTML_TEMPLATE_PATH = Path(__file__).resolve().parent / "web_ui" / "index.html"


def load_html_page() -> str:
    return HTML_TEMPLATE_PATH.read_text(encoding="utf-8")


class LLMClient:
    def __init__(self, model_name: str, base_url: str, api_key: str, temperature: float, max_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=300.0, max_retries=3)

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        messages = self._build_messages(user_prompt=user_prompt, system_prompt=system_prompt)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def generate_stream(self, user_prompt: str, system_prompt: str = "") -> Iterator[str]:
        messages = self._build_messages(user_prompt=user_prompt, system_prompt=system_prompt)
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1,
                stop=None,
                stream=True,
            )
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece is None:
                    continue
                piece_text = str(piece)
                if piece_text:
                    yield piece_text
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"llm_stream_error: {exc}") from exc

    @staticmethod
    def _build_messages(user_prompt: str, system_prompt: str) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = {
        "llm": {
            "model_name": args.model,
            "base_url": args.base_url,
            "api_key": args.api_key,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "system_prompt": args.system_prompt,
        },
        "env": {
            "seed": args.seed,
            "max_steps": args.max_steps,
            "history_length": args.history_length,
            "memory_type": args.memory_type,
            "enable_python_code": args.enable_python_code,
            "enable_local_rag": args.enable_local_rag,
            "python_code_timeout": args.python_code_timeout,
            "log_requests": args.log_requests,
        },
        "demo": {
            "ground_truth": args.ground_truth,
            "data_source": args.data_source,
        },
        "server": {
            "host": args.host,
            "port": args.port,
        },
        "runtime": {
            "max_conversations": args.max_conversations,
            "max_turns_per_conversation": args.max_turns_per_conversation,
        },
    }

    if args.config:
        file_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.create(file_cfg), OmegaConf.create(cfg)), resolve=True)

    if not cfg["llm"]["api_key"]:
        cfg["llm"]["api_key"] = os.environ.get("OPENAI_API_KEY", "EMPTY")

    if not cfg["llm"]["base_url"]:
        cfg["llm"]["base_url"] = "http://localhost:8000/v1"

    cfg["runtime"]["max_conversations"] = max(1, int(cfg["runtime"].get("max_conversations", 200)))
    cfg["runtime"]["max_turns_per_conversation"] = max(2, int(cfg["runtime"].get("max_turns_per_conversation", 20)))
    cfg["env"]["max_steps"] = max(1, int(cfg["env"].get("max_steps", 6)))
    cfg["llm"]["max_tokens"] = max(1, int(cfg["llm"].get("max_tokens", 4096)))

    return cfg


def build_manager(cfg: Dict[str, Any]) -> InformalMathTrainingEnvironmentManager:
    env_runtime_cfg = OmegaConf.create(
        {
            "max_steps": cfg["env"]["max_steps"],
            "informal_math_training": {
                "enable_python_code": cfg["env"]["enable_python_code"],
                "enable_local_rag": cfg["env"]["enable_local_rag"],
                "python_code_timeout": cfg["env"]["python_code_timeout"],
                "log_requests": cfg["env"]["log_requests"],
            },
        }
    )

    envs = build_informal_math_training_envs(
        seed=cfg["env"]["seed"],
        env_num=1,
        group_n=1,
        is_train=False,
        env_config=env_runtime_cfg,
    )

    manager_cfg = OmegaConf.create(
        {
            "env": {
                "env_name": "informal_math_training",
                "history_length": cfg["env"]["history_length"],
                "max_steps": cfg["env"]["max_steps"],
                "informal_math": {
                    "memory_type": cfg["env"]["memory_type"],
                    "enable_python_code": cfg["env"]["enable_python_code"],
                    "enable_local_rag": cfg["env"]["enable_local_rag"],
                    "execution_mode": "agentic",
                },
            }
        }
    )

    return InformalMathTrainingEnvironmentManager(envs, informal_math_training_projection, manager_cfg)


class DemoRuntime:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.logger = logging.getLogger("alphaapollo.demo.runtime")
        self.client = LLMClient(
            model_name=cfg["llm"]["model_name"],
            base_url=cfg["llm"]["base_url"],
            api_key=cfg["llm"]["api_key"],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_tokens"],
        )
        self.manager = build_manager(cfg)
        self.lock = threading.Lock()
        self.conversations: OrderedDict[str, list[Dict[str, str]]] = OrderedDict()
        self.max_conversations = int(cfg["runtime"]["max_conversations"])
        self.max_turns_per_conversation = int(cfg["runtime"]["max_turns_per_conversation"])

    @staticmethod
    def _extract_action_blocks(action: str) -> list[Dict[str, Any]]:
        blocks: list[Dict[str, Any]] = []
        if not action:
            return blocks

        pattern = re.compile(r"<(think|python_code|local_rag|answer)>(.*?)</\1>", re.DOTALL)
        cursor = 0

        for match in pattern.finditer(action):
            head = action[cursor : match.start()].strip()
            if head:
                blocks.append({"kind": "answer", "content": head})

            tag = match.group(1)
            content = match.group(2).strip()
            if content:
                if tag == "think":
                    blocks.append({"kind": "think", "content": content})
                elif tag == "answer":
                    blocks.append({"kind": "answer", "content": content})
                else:
                    blocks.append({"kind": "tool_call", "tool": tag, "content": content})

            cursor = match.end()

        tail = action[cursor:].strip()
        if tail:
            blocks.append({"kind": "answer", "content": tail})

        if not blocks and action.strip():
            sanitized = re.sub(r"</?(think|python_code|local_rag|answer)>", "", action, flags=re.IGNORECASE).strip()
            blocks.append({"kind": "answer", "content": sanitized or action.strip()})
        return blocks

    @staticmethod
    def _extract_tool_response(feedback: str) -> str:
        if not isinstance(feedback, str):
            return str(feedback)
        matches = re.findall(r"<tool_response>(.*?)</tool_response>", feedback, re.DOTALL)
        if matches:
            chunks = [m.strip() for m in matches if m and m.strip()]
            return "\n\n".join(chunks)
        cleaned = re.sub(r"</?tool_response>", "", feedback, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _extract_assistant_reply(action: str, blocks: list[Dict[str, Any]]) -> str:
        answer_texts = [b.get("content", "") for b in blocks if b.get("kind") == "answer" and b.get("content")]
        if answer_texts:
            return "\n\n".join(answer_texts).strip()
        return action.strip() if action else ""

    def _prune_conversations(self) -> None:
        while len(self.conversations) > self.max_conversations:
            self.conversations.popitem(last=False)

    def _append_turn(self, conv_id: str, role: str, content: str) -> None:
        history = self.conversations.get(conv_id, [])
        history.append({"role": role, "content": content})
        if len(history) > self.max_turns_per_conversation:
            history = history[-self.max_turns_per_conversation :]
        self.conversations[conv_id] = history
        self.conversations.move_to_end(conv_id)

    def solve(self, question: str) -> Dict[str, Any]:
        with self.lock:
            self.logger.info("solve_start question_len=%d", len(question))
            reset_kwargs = [
                {
                    "question": question,
                    "ground_truth": self.cfg["demo"]["ground_truth"],
                    "gt_traj": "",
                    "data_source": self.cfg["demo"]["data_source"],
                }
            ]

            obs, _ = self.manager.reset(kwargs=reset_kwargs)
            done = False
            step_id = 0
            reward = 0.0
            steps = []

            while not done and step_id < self.cfg["env"]["max_steps"]:
                step_id += 1
                prompt_text = obs["text"][0]
                action = self.client.generate(prompt_text, self.cfg["llm"]["system_prompt"])

                obs, rewards, dones, infos = self.manager.step([action])
                reward = float(rewards[0])
                done = bool(dones[0])
                info = infos[0] if infos else {}
                feedback = obs.get("anchor", [""])[0] if isinstance(obs, dict) else ""

                steps.append(
                    {
                        "step_id": step_id,
                        "prompt_text": prompt_text,
                        "action": action,
                        "feedback": feedback,
                        "is_action_valid": info.get("is_action_valid"),
                        "reward": reward,
                        "done": done,
                    }
                )

            self.logger.info("solve_done steps=%d final_reward=%s", step_id, reward)
            return {"steps": steps, "total_steps": step_id, "final_reward": reward}

    def solve_stream(self, question: str):
        with self.lock:
            self.logger.info("solve_stream_start question_len=%d", len(question))
            step_id = 0
            reward = 0.0
            try:
                reset_kwargs = [
                    {
                        "question": question,
                        "ground_truth": self.cfg["demo"]["ground_truth"],
                        "gt_traj": "",
                        "data_source": self.cfg["demo"]["data_source"],
                    }
                ]

                obs, _ = self.manager.reset(kwargs=reset_kwargs)
                done = False
                yield {"type": "start"}
                while not done and step_id < self.cfg["env"]["max_steps"]:
                    step_id += 1
                    prompt_text = obs["text"][0]
                    yield {"type": "step_start", "step_id": step_id, "prompt_text": prompt_text}

                    yield {"type": "llm_start", "step_id": step_id}
                    chunks: list[str] = []
                    for token in self.client.generate_stream(prompt_text, self.cfg["llm"]["system_prompt"]):
                        chunks.append(token)
                        yield {"type": "llm_token", "step_id": step_id, "token": token}
                    action = "".join(chunks).strip()
                    blocks = self._extract_action_blocks(action)
                    yield {"type": "llm_end", "step_id": step_id, "blocks": blocks}

                    obs, rewards, dones, infos = self.manager.step([action])
                    reward = float(rewards[0])
                    done = bool(dones[0])
                    info = infos[0] if infos else {}
                    feedback = obs.get("anchor", [""])[0] if isinstance(obs, dict) else ""
                    clean_feedback = self._extract_tool_response(feedback)
                    if clean_feedback:
                        yield {"type": "tool_response", "step_id": step_id, "content": clean_feedback}

                    yield {
                        "type": "step_end",
                        "step_id": step_id,
                        "done": done,
                        "reward": reward,
                        "is_action_valid": info.get("is_action_valid"),
                    }
                self.logger.info("solve_stream_done steps=%d final_reward=%s", step_id, reward)
                yield {"type": "done", "total_steps": step_id, "final_reward": reward}
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("solve_stream_error steps=%d", step_id)
                yield {"type": "error", "error": str(exc)}

    def chat_stream(self, message: str, conversation_id: str = "", reset: bool = False):
        with self.lock:
            conv_id = conversation_id.strip() if conversation_id else ""
            step_id = 0
            reward = 0.0
            assistant_reply = ""
            try:
                if reset or not conv_id or conv_id not in self.conversations:
                    conv_id = uuid.uuid4().hex[:12]
                    self.conversations[conv_id] = []
                self.conversations.move_to_end(conv_id)
                self._prune_conversations()

                self._append_turn(conv_id, "user", message)
                history = self.conversations[conv_id]
                prior = history[:-1][-8:]
                if prior:
                    transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in prior)
                    question = (
                        "You are continuing a multi-round user conversation.\n"
                        "Use prior context when relevant, but answer the latest request directly.\n\n"
                        f"Previous dialogue:\n{transcript}\n\n"
                        f"Current user message:\n{message}"
                    )
                else:
                    question = message

                self.logger.info(
                    "chat_stream_start conversation_id=%s reset=%s history_size=%d message_len=%d",
                    conv_id,
                    reset,
                    len(history),
                    len(message),
                )

                reset_kwargs = [
                    {
                        "question": question,
                        "ground_truth": self.cfg["demo"]["ground_truth"],
                        "gt_traj": "",
                        "data_source": self.cfg["demo"]["data_source"],
                    }
                ]

                obs, _ = self.manager.reset(kwargs=reset_kwargs)
                done = False
                yield {"type": "start", "conversation_id": conv_id, "user_message": message}
                while not done and step_id < self.cfg["env"]["max_steps"]:
                    step_id += 1
                    prompt_text = obs["text"][0]
                    yield {"type": "step_start", "step_id": step_id, "prompt_text": prompt_text}

                    yield {"type": "llm_start", "step_id": step_id}
                    chunks: list[str] = []
                    for token in self.client.generate_stream(prompt_text, self.cfg["llm"]["system_prompt"]):
                        chunks.append(token)
                        yield {"type": "llm_token", "step_id": step_id, "token": token}

                    action = "".join(chunks).strip()
                    blocks = self._extract_action_blocks(action)
                    assistant_reply = self._extract_assistant_reply(action, blocks)
                    yield {"type": "llm_end", "step_id": step_id, "blocks": blocks}

                    obs, rewards, dones, infos = self.manager.step([action])
                    reward = float(rewards[0])
                    done = bool(dones[0])
                    info = infos[0] if infos else {}
                    feedback = obs.get("anchor", [""])[0] if isinstance(obs, dict) else ""
                    clean_feedback = self._extract_tool_response(feedback)
                    if clean_feedback:
                        yield {"type": "tool_response", "step_id": step_id, "content": clean_feedback}

                    yield {
                        "type": "step_end",
                        "step_id": step_id,
                        "done": done,
                        "reward": reward,
                        "is_action_valid": info.get("is_action_valid"),
                    }

                if assistant_reply:
                    self._append_turn(conv_id, "assistant", assistant_reply)
                self.logger.info(
                    "chat_stream_done conversation_id=%s steps=%d final_reward=%s",
                    conv_id,
                    step_id,
                    reward,
                )
                yield {
                    "type": "done",
                    "conversation_id": conv_id,
                    "total_steps": step_id,
                    "final_reward": reward,
                    "assistant_reply": assistant_reply,
                }
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("chat_stream_error conversation_id=%s steps=%d", conv_id, step_id)
                yield {"type": "error", "conversation_id": conv_id, "error": str(exc)}


RUNTIME: DemoRuntime | None = None


def _json_default(value: Any) -> Any:
    # Convert numpy/pandas-like objects without importing optional deps here.
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


class DemoHandler(BaseHTTPRequestHandler):
    _STREAM_EVENT_REQUIRED: Dict[str, tuple[str, ...]] = {
        "start": (),
        "step_start": ("step_id", "prompt_text"),
        "llm_start": ("step_id",),
        "llm_token": ("step_id", "token"),
        "llm_end": ("step_id", "blocks"),
        "tool_response": ("step_id", "content"),
        "step_end": ("step_id", "done", "reward", "is_action_valid"),
        "done": ("total_steps", "final_reward"),
        "error": ("error",),
    }

    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        body = json.dumps(data, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, error: str, detail: str = "") -> None:
        payload: Dict[str, Any] = {"error": error}
        if detail:
            payload["detail"] = detail
        self._send_json(status, payload)

    @classmethod
    def _normalize_stream_event(cls, raw_event: Any, is_chat_stream: bool) -> Dict[str, Any]:
        if not isinstance(raw_event, Mapping):
            return {"type": "error", "error": "invalid_event: non-object"}

        event: Dict[str, Any] = dict(raw_event)
        event_type = str(event.get("type", "")).strip()
        if not event_type:
            return {"type": "error", "error": "invalid_event: missing type"}
        if event_type not in cls._STREAM_EVENT_REQUIRED:
            return {"type": "error", "error": f"invalid_event: unsupported type {event_type}"}

        normalized: Dict[str, Any] = {"type": event_type}
        normalized.update(event)

        if event_type == "start":
            if is_chat_stream:
                normalized.setdefault("conversation_id", "")
                normalized.setdefault("user_message", "")
        elif event_type == "step_start":
            normalized.setdefault("prompt_text", "")
        elif event_type == "llm_token":
            normalized.setdefault("token", "")
        elif event_type == "llm_end":
            blocks = normalized.get("blocks", [])
            normalized["blocks"] = blocks if isinstance(blocks, list) else []
        elif event_type == "tool_response":
            normalized.setdefault("content", "")
        elif event_type == "step_end":
            normalized.setdefault("done", False)
            normalized.setdefault("reward", 0.0)
            normalized.setdefault("is_action_valid", None)
        elif event_type == "done":
            normalized.setdefault("total_steps", 0)
            normalized.setdefault("final_reward", 0.0)
            if is_chat_stream:
                normalized.setdefault("conversation_id", "")
                normalized.setdefault("assistant_reply", "")
        elif event_type == "error":
            normalized["error"] = str(normalized.get("error", "unknown_error"))

        missing = [key for key in cls._STREAM_EVENT_REQUIRED[event_type] if key not in normalized]
        if missing:
            return {"type": "error", "error": f"invalid_event: missing fields {', '.join(missing)}"}
        return normalized

    def _send_sse_event(self, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, default=_json_default)
        self.wfile.write(f"event: message\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        logger = logging.getLogger("alphaapollo.demo.http")
        started = time.time()
        if self.path in {"/", "/index.html"}:
            body = load_html_page().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            logger.info("request method=GET path=%s status=200 duration_ms=%d", self.path, int((time.time() - started) * 1000))
            return

        if self.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"ok": True})
            logger.info("request method=GET path=%s status=200 duration_ms=%d", self.path, int((time.time() - started) * 1000))
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "not_found")
        logger.info("request method=GET path=%s status=404 duration_ms=%d", self.path, int((time.time() - started) * 1000))

    def do_POST(self) -> None:  # noqa: N802
        logger = logging.getLogger("alphaapollo.demo.http")
        started = time.time()
        if self.path not in {"/api/solve", "/api/solve/stream", "/api/chat/stream"}:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not_found")
            logger.info("request method=POST path=%s status=404 duration_ms=%d", self.path, int((time.time() - started) * 1000))
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)

        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid_json")
            logger.info("request method=POST path=%s status=400 error=invalid_json duration_ms=%d", self.path, int((time.time() - started) * 1000))
            return

        question = str(payload.get("question", "")).strip()
        message = str(payload.get("message", "")).strip()
        if self.path == "/api/chat/stream":
            if not message:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "message is required", "invalid_request")
                logger.info("request method=POST path=%s status=400 error=message_required duration_ms=%d", self.path, int((time.time() - started) * 1000))
                return
        else:
            if not question:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "question is required", "invalid_request")
                logger.info("request method=POST path=%s status=400 error=question_required duration_ms=%d", self.path, int((time.time() - started) * 1000))
                return

        if self.path in {"/api/solve/stream", "/api/chat/stream"}:
            is_chat_stream = self.path == "/api/chat/stream"
            conversation_id = str(payload.get("conversation_id", "")).strip() if is_chat_stream else ""
            try:
                assert RUNTIME is not None
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                event_count = 0
                terminal_type = ""
                if is_chat_stream:
                    reset = bool(payload.get("reset", False))
                    stream_iter = RUNTIME.chat_stream(
                        message=message,
                        conversation_id=conversation_id,
                        reset=reset,
                    )
                else:
                    stream_iter = RUNTIME.solve_stream(question)

                for event in stream_iter:
                    normalized_event = self._normalize_stream_event(event, is_chat_stream=is_chat_stream)
                    self._send_sse_event(normalized_event)
                    event_count += 1
                    etype = normalized_event.get("type")
                    if etype in {"done", "error"}:
                        terminal_type = str(etype)
                logger.info(
                    "request method=POST path=%s status=200 stream=true conversation_id=%s events=%d terminal=%s duration_ms=%d",
                    self.path,
                    conversation_id,
                    event_count,
                    terminal_type or "none",
                    int((time.time() - started) * 1000),
                )
            except (BrokenPipeError, ConnectionResetError):
                logger.info(
                    "request method=POST path=%s status=200 stream=true disconnected=true conversation_id=%s duration_ms=%d",
                    self.path,
                    conversation_id,
                    int((time.time() - started) * 1000),
                )
                return
            except Exception as exc:  # noqa: BLE001
                try:
                    error_event = self._normalize_stream_event(
                        {"type": "error", "error": str(exc)},
                        is_chat_stream=is_chat_stream,
                    )
                    self._send_sse_event(error_event)
                except Exception:  # noqa: BLE001
                    return
                logger.exception(
                    "request method=POST path=%s status=500 stream=true conversation_id=%s duration_ms=%d",
                    self.path,
                    conversation_id,
                    int((time.time() - started) * 1000),
                )
            return

        try:
            assert RUNTIME is not None
            result = RUNTIME.solve(question)
            self._send_json(HTTPStatus.OK, result)
            logger.info(
                "request method=POST path=%s status=200 stream=false steps=%s duration_ms=%d",
                self.path,
                result.get("total_steps"),
                int((time.time() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", str(exc))
            logger.exception(
                "request method=POST path=%s status=500 stream=false duration_ms=%d",
                self.path,
                int((time.time() - started) * 1000),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web demo for informal_math_training.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--system-prompt", type=str, default="")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--memory-type", choices=["simple", "score", "ndimensional"], default="simple")
    parser.add_argument("--enable-python-code", action="store_true", default=True)
    parser.add_argument("--disable-python-code", dest="enable_python_code", action="store_false")
    parser.add_argument("--enable-local-rag", action="store_true", default=False)
    parser.add_argument("--python-code-timeout", type=int, default=30)
    parser.add_argument("--log-requests", action="store_true", default=False)

    parser.add_argument("--ground-truth", type=str, default="\\boxed{0}")
    parser.add_argument("--data-source", type=str, default="web_demo")
    parser.add_argument("--max-conversations", type=int, default=200)
    parser.add_argument("--max-turns-per-conversation", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    global RUNTIME

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    args = parse_args()
    cfg = merge_config(args)
    RUNTIME = DemoRuntime(cfg)

    host = cfg["server"]["host"]
    port = int(cfg["server"]["port"])
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Web demo listening at http://{host}:{port}")
    print(f"Using model={cfg['llm']['model_name']} base_url={cfg['llm']['base_url']}")

    try:
        server.serve_forever()
    finally:
        if RUNTIME is not None:
            RUNTIME.manager.envs.close()


if __name__ == "__main__":
    main()
