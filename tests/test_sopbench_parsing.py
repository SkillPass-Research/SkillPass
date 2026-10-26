from skillpass.sopbench.domain import final_answer, lenient_match, strict_match, tool_calls
from skillpass.sopbench.runner import default_decoys, procedure_family


def test_tool_calls_reads_the_first_json_object_after_each_tag():
    reply = (
        "I will look it "
        'up.\n<tool_call>\n{"name": "get_case", "arguments": {"id": "C-1"}}\n</tool_call>'
        '<tool_call>\n{"name": "broken", "arguments": {'
    )
    calls = tool_calls(reply)
    assert calls[0] == {"name": "get_case", "arguments": {"id": "C-1"}}
    assert "_unparsed" in calls[1]
    assert tool_calls("no calls here") == []


def test_final_answer_takes_the_last_answer_under_any_tag():
    reply = (
        '<final_output>{"decision": "approve"}</final_output> ... on reflection '
        '<final_decision>{"decision": "reject"}</final_decision>'
    )
    assert final_answer(reply, ("final_output", "final_decision")) == {"decision": "reject"}
    assert final_answer('final_output={"decision": "approve"}') == {"decision": "approve"}
    assert final_answer("<final_output>approve</final_output>", fields=["decision"]) == {
        "decision": "approve"
    }
    assert final_answer("nothing") is None


def test_strict_match_ignores_case_and_formatting_but_not_content():
    assert strict_match({"decision": "**Approve**."}, {"decision": "approve"})
    assert not strict_match({"decision": "approved"}, {"decision": "approve"})
    assert lenient_match({"decision": "approved"}, {"decision": "approve"})  # SOP-Bench's own rule
    assert not strict_match(None, {"decision": "approve"})


def test_default_decoys_skip_versions_of_the_same_procedure():
    assert procedure_family("referral_abuse_detection_v2") == "referral_abuse_detection"
    domains = ["a", "referral_abuse_detection_v1", "referral_abuse_detection_v2", "b"]
    decoys = default_decoys(domains)
    assert decoys["a"] == "referral_abuse_detection_v1"
    assert decoys["referral_abuse_detection_v1"] == "b"
    assert decoys["referral_abuse_detection_v2"] == "b"
    assert decoys["b"] == "a"
