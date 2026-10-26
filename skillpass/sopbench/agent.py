"""The tool-calling agent (Section 5.1, *Skill representation and harness*).

The conversation opens with a system message describing the domain's tools and how to call
them, the same for every presentation and reader. The first user message is the `text`
presentation's prompt: the reading instruction, the procedure, then the case; `none` has the
case alone; `plain` and `spass` are the `text` conversation with the tensor in the procedure's
place, through `Reader.chat_many`. Each reply's tool calls are run and their results returned
as the next user message; an episode ends at a reply with no tool call, or after `MAX_TURNS`
turns of `TURN_TOKENS` tokens each. Its answer is the last one under an answer tag.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from skillpass.prompts import SEPARATOR, TASK_PREFIX, TEXT_INSTRUCTION
from skillpass.registry import AGENT_PROBES, Presentation
from skillpass.sopbench.domain import (
    TOOL_RESPONSE_TAG,
    Domain,
    Task,
    ToolBox,
    final_answer,
    lenient_match,
    strict_match,
    task_text,
    tool_calls,
)

MAX_TURNS = 10
TURN_TOKENS = 1024
BATCH = 16
"""Episodes advanced together; each turn is one batched generation of at most this many."""

SYSTEM = (
    "You are an agent that completes tasks by calling tools. To call a tool, reply with one or "
    'more blocks of the form\n<tool_call>\n{"name": <tool name>, "arguments": {<arguments>}}'
    "\n</tool_call>\nand nothing after them; each result comes back in a <tool_response> block. "
    "When you have your final answer, reply with it and no tool call.\n\nAvailable tools:\n"
)


def system_message(domain: Domain) -> str:
    return SYSTEM + json.dumps(domain.tools, indent=1)


def first_message(presentation: Presentation, task: Task, domain: Domain) -> str:
    question = TASK_PREFIX + task_text(task)
    if presentation is Presentation.NONE:
        return question
    return TEXT_INSTRUCTION + SEPARATOR + domain.sop + SEPARATOR + question


def start_messages(presentation: Presentation, task: Task, domain: Domain) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_message(domain)},
        {"role": "user", "content": first_message(presentation, task, domain)},
    ]


def agent_conversations(domain: Domain) -> list[list[dict[str, str]]]:
    """The synthetic agent conversations an SOP-Bench S-Pass is fitted under beside the generic
    probes (Section 3, *probe contexts*): each `AGENT_PROBES` turn after the domain's own system
    message and the `text` presentation's first message, exactly as an episode begins. The
    cases are invented; only the system message and the procedure are the domain's."""
    conversations = []
    for i, probe in enumerate(AGENT_PROBES):
        task = Task(f"probe-{i}", dict(probe.inputs), dict.fromkeys(probe.fields, ""))
        messages = start_messages(Presentation.TEXT, task, domain)
        if probe.exchange is not None:
            call, response = probe.exchange
            messages += [
                {"role": "assistant", "content": call},
                {"role": "user", "content": response},
            ]
        conversations.append(messages)
    return conversations


def run_tools(toolbox: ToolBox, reply: str, made: list[dict[str, Any]] | None = None) -> str | None:
    """The `<tool_response>` blocks for the reply's calls, or `None` when it makes none. A tool's
    refusal goes back to the agent as an error."""
    calls = tool_calls(reply)
    if not calls:
        return None
    out = []
    for call in calls:
        arguments = call.get("arguments") if isinstance(call, dict) else None
        error = None
        try:
            if "_unparsed" in call or "name" not in call or not isinstance(arguments, dict):
                raise ValueError("the tool call is not a JSON object with a name and arguments")
            result = toolbox.call(call["name"], dict(arguments))
        except Exception as exc:  # noqa: BLE001 - the agent sees the tool's own refusal
            error = f"{type(exc).__name__}: {exc}"
            result = {"error": error}
        if made is not None:
            made.append(
                {
                    "name": str(call.get("name", "")),
                    "arguments": dict(arguments) if isinstance(arguments, dict) else None,
                    "result": result,
                    "error": error,
                }
            )
        out.append(f"{TOOL_RESPONSE_TAG}\n{json.dumps(result, default=str)}\n</tool_response>")
    return "\n".join(out)


@dataclass
class Episode:
    task: Task
    messages: list[dict[str, str]]
    domain: Domain
    turns: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False

    def record(self) -> dict[str, Any]:
        final = None
        for turn in reversed(self.turns):
            final = final_answer(turn["reply"], self.domain.answer_tags, list(self.task.expected))
            if final is not None:
                break
        return {
            "task": self.task.id,
            "expected": self.task.expected,
            "final": final,
            "passed": strict_match(final, self.task.expected),
            "lenient": lenient_match(final, self.task.expected),
            "turns": len(self.turns),
            "tool_calls": len(self.calls),
            "finished": self.finished,
            "trajectory": self.turns,
        }


def run_episodes(
    reader,
    presentation: Presentation,
    tasks: Sequence[Task],
    domain: Domain,
    toolbox: ToolBox,
    tensor=None,
    *,
    max_turns: int = MAX_TURNS,
    turn_tokens: int = TURN_TOKENS,
) -> list[dict[str, Any]]:
    """Every task's episode under `presentation`, advanced a turn at a time as one batch;
    `tensor` is the `plain` or `spass` tensor and `None` otherwise."""
    injected = presentation in (Presentation.PLAIN, Presentation.SPASS)
    if injected and tensor is None:
        raise ValueError(f"the {presentation.value} presentation needs its tensor")
    episodes = [Episode(task, start_messages(presentation, task, domain), domain) for task in tasks]
    for _ in range(max_turns):
        live = [e for e in episodes if not e.finished]
        if not live:
            break
        items = [
            (e.messages, tensor if injected else None, domain.sop if injected else None)
            for e in live
        ]
        replies = reader.chat_many(items, max_new_tokens=turn_tokens)
        for episode, reply in zip(live, replies, strict=True):
            reply = reply.split(TOOL_RESPONSE_TAG)[0]  # a result the model writes is not one
            episode.messages.append({"role": "assistant", "content": reply})
            response = run_tools(toolbox, reply, episode.calls)
            episode.turns.append({"reply": reply, "tools": response})
            if response is None:
                episode.finished = True
            else:
                episode.messages.append({"role": "user", "content": response})
    return [e.record() for e in episodes]
