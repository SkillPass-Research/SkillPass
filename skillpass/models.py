"""The readers: every model behind one interface, through Hugging Face `transformers`.

Three calls. `read_many(prompts)` is the token path: each prompt is one user message laid out
by the tokenizer's chat template. `read_embeds_many(items)` is the injection path: the same
`text` prompt, templated the same way, with the text before and after the skill embedded
through the model's own table and the tensor spliced in where the skill's text would have
been embedded. `chat_many(items)` is the same splice inside a whole conversation, which is how
the SOP-Bench agent reads an S-Pass. Decoding is greedy. Nothing heavy is imported until a
model loads, so every reader can be built on a machine without `torch`.

A reader built with `blocks=k` loads only the embedding table and the first `k` blocks: all the
construction reads, which is how three relatives sit beside the authorized model on one GPU.
Gemma scales its embeddings inside the model; the injection path takes the embedding module's
output around the skill, so a tensor for Gemma must be in those scaled coordinates
(`spass.embed.table_of`).

`SKILLPASS_DEVICE_MAP` names a `transformers` device map (`auto` spreads a large model over the
visible GPUs; `cpu` runs a tiny model in float32 for a smoke test); by default each model goes
to `cuda` in bfloat16.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any, ClassVar

from skillpass import registry
from skillpass.prompts import split_prompt

DEVICE_MAP_VARIABLE = "SKILLPASS_DEVICE_MAP"
TOKEN_VARIABLE = "HF_TOKEN"
FAKE_FAMILY = "fake"
SPASS_MARK = "<spass>"


class ReaderError(RuntimeError):
    def __init__(self, reader_id: str, reason: str) -> None:
        self.reader_id, self.reason = reader_id, reason
        super().__init__(f"reader {reader_id!r}: {reason}")


def _device_map() -> str:
    return os.environ.get(DEVICE_MAP_VARIABLE) or "cuda"


def _token() -> str | None:
    return os.environ.get(TOKEN_VARIABLE) or None


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _eos_ids(eos: int | Sequence[int] | None) -> tuple[int, ...]:
    if eos is None:
        return ()
    return (eos,) if isinstance(eos, int) else tuple(eos)


def finish_reason(new_tokens: Sequence[int], eos_ids: Iterable[int], max_new_tokens: int) -> str:
    if new_tokens and new_tokens[-1] in set(eos_ids):
        return "stop"
    return "length" if len(new_tokens) >= max_new_tokens else "stop"


Messages = Sequence[Mapping[str, str]]


class ReaderBase:
    """The interface. `last_finish_reasons` holds `"stop"` or `"length"` per answer."""

    entry: registry.Reader

    def __init__(self) -> None:
        self.last_finish_reasons: list[str] = []

    def read_many(self, prompts: Sequence[str], *, max_new_tokens: int) -> list[str]:
        raise NotImplementedError

    def read_embeds_many(
        self, items: Sequence[tuple[Any, str, str]], *, max_new_tokens: int
    ) -> list[str]:
        """Each item is `(tensor, prompt, skill_text)`: the tensor takes `skill_text`'s place."""
        raise NotImplementedError

    def chat_many(
        self, items: Sequence[tuple[Messages, Any | None, str | None]], *, max_new_tokens: int
    ) -> list[str]:
        """Each item is `(messages, tensor, skill_text)`; `tensor` `None` is the token path."""
        raise NotImplementedError

    def read(self, prompt: str, *, max_new_tokens: int) -> str:
        return self.read_many([prompt], max_new_tokens=max_new_tokens)[0]

    def read_embeds(self, tensor: Any, prompt: str, skill_text: str, *, max_new_tokens: int) -> str:
        return self.read_embeds_many([(tensor, prompt, skill_text)], max_new_tokens=max_new_tokens)[
            0
        ]

    def identity(self) -> dict[str, str]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> ReaderBase:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class HFReader(ReaderBase):
    model_class: ClassVar[str]

    def __init__(self, entry: registry.Reader, *, blocks: int | None = None) -> None:
        if entry.revision is None:
            raise ReaderError(entry.reader_id, "no pinned revision in the registry entry")
        super().__init__()
        self.entry = entry
        self.blocks = blocks
        self._model: Any = None
        self._tokenizer: Any = None

    # loading

    def identity(self) -> dict[str, str]:
        return {
            "reader_id": self.entry.reader_id,
            "family": self.entry.family,
            "kind": "hf",
            "revision": self.entry.revision or "",
            "transformers": _version("transformers"),
            "torch": _version("torch"),
            "device": _device_map(),
        }

    @property
    def model(self) -> Any:
        self.load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        self.load()
        return self._tokenizer

    def load(self) -> None:
        if self._model is not None:
            return
        reader_id, revision = self.entry.reader_id, self.entry.revision
        try:
            import torch
            import transformers
        except ImportError as error:
            raise ReaderError(
                reader_id, f"needs the gpu group (uv sync --group gpu): {error}"
            ) from error
        device_map = _device_map()
        if device_map == "cuda" and not torch.cuda.is_available():
            raise ReaderError(reader_id, "needs a CUDA GPU, and torch sees none")
        dtype = torch.float32 if device_map == "cpu" else torch.bfloat16
        token = _token()
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                reader_id, revision=revision, token=token
            )
            if getattr(tokenizer, "chat_template", None) is None and self.entry.chat_template_from:
                lender = registry.ALL_READERS[self.entry.chat_template_from]
                tokenizer.chat_template = transformers.AutoTokenizer.from_pretrained(
                    lender.reader_id, revision=lender.revision, token=token
                ).chat_template
            extra = {}
            if self.blocks is not None:
                config = transformers.AutoConfig.from_pretrained(
                    reader_id, revision=revision, token=token
                )
                text = config.get_text_config()
                if not 1 <= self.blocks <= text.num_hidden_layers:
                    raise ValueError(f"blocks must be in 1..{text.num_hidden_layers}")
                text.num_hidden_layers = self.blocks
                if getattr(text, "layer_types", None):
                    text.layer_types = list(text.layer_types)[: self.blocks]
                extra["config"] = config
            model = getattr(transformers, self.model_class).from_pretrained(
                reader_id,
                revision=revision,
                token=token,
                dtype=dtype,
                device_map=device_map,
                **extra,
            )
        except Exception as error:
            raise ReaderError(reader_id, f"could not load: {error}") from error
        model.eval()
        generation = model.generation_config
        generation.do_sample = False
        generation.temperature = None
        generation.top_p = None
        generation.top_k = None
        self._tokenizer, self._model = tokenizer, model

    def close(self) -> None:
        if self._model is None:
            return
        self._model = self._tokenizer = None
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # templates

    def template(self, prompt: str) -> str:
        """The chat-templated text of one user message holding `prompt`."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )

    def render(self, messages: Messages) -> str:
        """A conversation as this model reads it, ready for the next assistant turn."""
        return self.tokenizer.apply_chat_template(
            [dict(m) for m in messages], add_generation_prompt=True, tokenize=False
        )

    def ids(self, text: str) -> Any:
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]

    def splice_ids(self, prompt: str, skill_text: str) -> tuple[Any, Any]:
        """The token ids before and after the skill's place in the templated `text` prompt."""
        before, after = split_prompt(prompt, skill_text)
        text = self.template(before + skill_text + after)
        if text.count(skill_text) != 1:
            raise ReaderError(self.entry.reader_id, "the template did not keep the skill text once")
        pre, post = text.split(skill_text, 1)
        return self.ids(pre), self.ids(post)

    # the three paths

    def read_many(self, prompts: Sequence[str], *, max_new_tokens: int) -> list[str]:
        if not prompts:
            self.last_finish_reasons = []
            return []
        embed = self.model.get_input_embeddings()
        device = self.model.device
        rows = [embed(self.ids(self.template(p)).to(device)) for p in prompts]
        return self._generate(rows, max_new_tokens)

    def _splice(self, tensor: Any, pre_ids: Any, post_ids: Any) -> Any:
        import numpy as np
        import torch

        embed = self.model.get_input_embeddings()
        device = self.model.device
        array = np.asarray(tensor, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != embed.weight.shape[1]:
            raise ReaderError(
                self.entry.reader_id,
                f"a tensor of shape {array.shape} does not fit width {embed.weight.shape[1]}",
            )
        span = torch.tensor(array, device=device, dtype=embed.weight.dtype)
        return torch.cat([embed(pre_ids.to(device)), span, embed(post_ids.to(device))], dim=0)

    def read_embeds_many(
        self, items: Sequence[tuple[Any, str, str]], *, max_new_tokens: int
    ) -> list[str]:
        if not items:
            self.last_finish_reasons = []
            return []
        rows = []
        for tensor, prompt, skill_text in items:
            pre, post = self.splice_ids(prompt, skill_text)
            rows.append(self._splice(tensor, pre, post))
        return self._generate(rows, max_new_tokens)

    def chat_many(
        self, items: Sequence[tuple[Messages, Any | None, str | None]], *, max_new_tokens: int
    ) -> list[str]:
        if not items:
            self.last_finish_reasons = []
            return []
        embed = self.model.get_input_embeddings()
        device = self.model.device
        rows = []
        for messages, tensor, skill_text in items:
            text = self.render(messages)
            if tensor is None:
                rows.append(embed(self.ids(text).to(device)))
                continue
            if skill_text is None or text.count(skill_text) != 1:
                raise ReaderError(
                    self.entry.reader_id, "the conversation must hold the skill text exactly once"
                )
            pre, post = text.split(skill_text, 1)
            rows.append(self._splice(tensor, self.ids(pre), self.ids(post)))
        return self._generate(rows, max_new_tokens)

    def _generate(self, rows: Sequence[Any], max_new_tokens: int) -> list[str]:
        """Greedy continuations of `rows`, each `(n_i, d)` input embeddings, as one left-padded
        batch; one answer and one finish reason per row."""
        import torch

        model, tokenizer = self.model, self.tokenizer
        device, width = model.device, rows[0].shape[1]
        longest = max(row.shape[0] for row in rows)
        embeds = torch.zeros((len(rows), longest, width), dtype=rows[0].dtype, device=device)
        attention = torch.zeros((len(rows), longest), dtype=torch.long, device=device)
        for i, row in enumerate(rows):
            embeds[i, longest - row.shape[0] :] = row
            attention[i, longest - row.shape[0] :] = 1
        eos = _eos_ids(model.generation_config.eos_token_id)
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = eos[0] if eos else 0
        _forward_embeds(model)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*repetition_penalty.*inputs_embeds.*")
            output = model.generate(
                inputs_embeds=embeds,
                attention_mask=attention,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad,
            )
        answers, reasons = [], []
        for new_tokens in output.tolist():  # with inputs_embeds, generate returns new tokens only
            cut = next((i + 1 for i, t in enumerate(new_tokens) if t in eos), len(new_tokens))
            kept = new_tokens[:cut]
            reasons.append(finish_reason(kept, eos, max_new_tokens))
            answers.append(tokenizer.decode(kept, skip_special_tokens=True))
        self.last_finish_reasons = reasons
        return answers


class CausalLMReader(HFReader):
    model_class = "AutoModelForCausalLM"


class Qwen25VLReader(HFReader):
    """The language side of Qwen2.5-VL, read with text alone."""

    model_class = "Qwen2_5_VLForConditionalGeneration"


class Gemma3Reader(HFReader):
    model_class = "Gemma3ForConditionalGeneration"


class Gemma4UnifiedReader(HFReader):
    model_class = "Gemma4UnifiedForConditionalGeneration"


def _forward_embeds(model: Any) -> None:
    """Let `generate` take `inputs_embeds` on a model whose generation hook only passes them on
    through `**kwargs` (Gemma 3 and 4)."""
    import inspect

    hook = model.prepare_inputs_for_generation
    if "inputs_embeds" in inspect.signature(hook).parameters:
        return

    def prepare_inputs_for_generation(input_ids, inputs_embeds=None, **kwargs):
        return hook(input_ids, inputs_embeds=inputs_embeds, **kwargs)

    model.prepare_inputs_for_generation = prepare_inputs_for_generation


READERS: dict[str, type[HFReader]] = {
    "causal-lm": CausalLMReader,
    "qwen2.5-vl": Qwen25VLReader,
    "gemma3": Gemma3Reader,
    "gemma4-unified": Gemma4UnifiedReader,
}


@dataclass(frozen=True)
class Call:
    prompt: str
    max_new_tokens: int
    tensor_shape: tuple[int, ...] | None = None


class FakeReader(ReaderBase):
    """A scripted reader for tests and dry runs: the first script key that is a substring of the
    prompt gives the answer, `default` answers the rest. Never evidence."""

    def __init__(self, entry: registry.Reader, script: Mapping[str, str] | None) -> None:
        if script is None or "default" not in script:
            raise ReaderError(entry.reader_id, "a fake reader's script needs a 'default' answer")
        super().__init__()
        self.entry = entry
        self.script = dict(script)
        self.calls: list[Call] = []

    def _answer(self, prompt: str, max_new_tokens: int, shape: tuple[int, ...] | None) -> str:
        self.calls.append(Call(prompt, max_new_tokens, shape))
        for key, answer in self.script.items():
            if key != "default" and key in prompt:
                return answer
        return self.script["default"]

    def read_many(self, prompts: Sequence[str], *, max_new_tokens: int) -> list[str]:
        self.last_finish_reasons = ["stop"] * len(prompts)
        return [self._answer(p, max_new_tokens, None) for p in prompts]

    def read_embeds_many(self, items, *, max_new_tokens: int) -> list[str]:
        import numpy as np

        out = []
        for tensor, prompt, skill_text in items:
            marked = prompt.replace(skill_text, SPASS_MARK, 1)
            out.append(self._answer(marked, max_new_tokens, tuple(np.asarray(tensor).shape)))
        self.last_finish_reasons = ["stop"] * len(items)
        return out

    def chat_many(self, items, *, max_new_tokens: int) -> list[str]:
        import numpy as np

        out = []
        for messages, tensor, skill_text in items:
            prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
            shape = None
            if tensor is not None:
                prompt = prompt.replace(skill_text or "", SPASS_MARK, 1)
                shape = tuple(np.asarray(tensor).shape)
            out.append(self._answer(prompt, max_new_tokens, shape))
        self.last_finish_reasons = ["stop"] * len(items)
        return out

    def identity(self) -> dict[str, str]:
        return {"reader_id": self.entry.reader_id, "family": FAKE_FAMILY, "kind": "fake"}


def make_reader(
    entry: registry.Reader, *, blocks: int | None = None, script: Mapping[str, str] | None = None
) -> ReaderBase:
    """The wrapper for `entry`, built and not yet loaded."""
    if entry.family == FAKE_FAMILY:
        return FakeReader(entry, script)
    if entry.family not in READERS:
        raise ReaderError(entry.reader_id, f"no wrapper for family {entry.family!r}")
    return READERS[entry.family](entry, blocks=blocks)
