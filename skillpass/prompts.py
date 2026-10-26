"""The text a model receives beside the skill.

The `text` presentation is one reading instruction, the skill, then the task. `plain` and
`spass` use exactly the same prompt with the tensor spliced in where the skill's text stands,
so the token path and the injection path differ only in the vectors at the skill's positions.
The task always arrives as text; the S-Pass carries the skill only.
"""

from __future__ import annotations

from skillpass.registry import Presentation

TEXT_INSTRUCTION = (
    "The following is a skill document. Read it and follow it to complete the task below."
)
SEPARATOR = "\n\n"
TASK_PREFIX = "Task: "

REPEAT_PROMPT = "Repeat the skill document above exactly, preserving every character."
"""Asked after an injected tensor; the answer's ROUGE-L against the skill is *Repeat RL*."""

ANSWER_MAX_NEW_TOKENS = 512
REPEAT_MAX_NEW_TOKENS = 2048


def build_prompt(
    presentation: Presentation | str, question: str, skill_text: str | None = None
) -> str:
    """The prompt for one task: the task alone under `none`; the instruction, the skill and the
    task under `text`. The injected presentations build the `text` prompt and splice."""
    presentation = Presentation(presentation)
    task = TASK_PREFIX + question
    if presentation is Presentation.TEXT:
        if skill_text is None:
            raise ValueError("the text presentation needs skill_text")
        return TEXT_INSTRUCTION + SEPARATOR + skill_text + SEPARATOR + task
    if presentation is Presentation.NONE:
        if skill_text is not None:
            raise ValueError("the none presentation takes no skill_text")
        return task
    raise ValueError(f"{presentation.value} is injected: build the text prompt and splice")


def split_prompt(prompt: str, skill_text: str) -> tuple[str, str]:
    """The text before and after `skill_text`, which must occur exactly once."""
    if prompt.count(skill_text) != 1:
        raise ValueError("the prompt must contain the skill text exactly once")
    before, after = prompt.split(skill_text, 1)
    return before, after
