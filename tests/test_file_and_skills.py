import numpy as np
import pytest

from skillpass.skills import SUITE_ROOT, check, load_skill, load_suite
from skillpass.spass.file import read_plain, read_spass, write_plain, write_spass

SIDECAR = {
    "construction": "spass",
    "skill": "x",
    "model": "m",
    "revision": "r",
    "tokens": 3,
    "width": 4,
    "seed": 0,
    "decoy": "y",
    "source_sha256": "0" * 64,
}


def test_spass_roundtrip_is_float16(tmp_path):
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    write_spass(tmp_path, tensor, SIDECAR)
    write_plain(tmp_path, tensor)
    back = read_spass(tmp_path)
    assert back.tensor.dtype == np.float16
    assert np.allclose(back.tensor.astype(np.float32), tensor, atol=1e-3)
    assert back.sidecar["skill"] == "x"
    assert read_plain(tmp_path).shape == (3, 4)


def test_spass_refuses_a_sidecar_missing_a_field(tmp_path):
    with pytest.raises(ValueError):
        write_spass(
            tmp_path,
            np.zeros((3, 4), dtype=np.float32),
            {k: v for k, v in SIDECAR.items() if k != "decoy"},
        )
    with pytest.raises(ValueError):
        write_spass(tmp_path, np.zeros((2, 4), dtype=np.float32), SIDECAR)


def test_the_held_out_skills_load_with_ten_tasks_each():
    suite = load_suite(SUITE_ROOT)
    assert [s.name for s in suite] == ["cloudloom-stilling", "fictional-api", "vell-units"]
    for skill in suite:
        assert len(skill.tasks) == 10
        assert skill.transcription == " ".join(skill.source.split())
        assert all(i in skill.source for i in skill.identifiers)


def test_checkers_compare_content_not_formatting():
    skill = load_skill(SUITE_ROOT / "fictional-api")
    tasks = {t.id: t for t in skill.tasks}
    assert check(tasks["first-call"], "twf_greet")
    assert check(tasks["first-call"], "**twf_greet**.")
    assert check(tasks["first-call"], "```\ntwf_greet\n```")
    assert not check(tasks["first-call"], "twf_reserve_berth")
    assert check(tasks["overweight-error"], "TWF-240 CRATE_OVERHEFT")
    assert check(
        tasks["load-crate-parameters"], "It takes crate_glyph, heft_in_grains and the tidemark."
    )
    assert not check(tasks["load-crate-parameters"], "crate_glyph only")
