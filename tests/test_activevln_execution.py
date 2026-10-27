"""Continuation protocol validation and exact disabled-path prompt preservation."""

from types import SimpleNamespace

import pytest

from embodiinfer.policies.activevln.execution_activevln import EXECUTION_PREFIX_SCHEMA, ExecutionPrefix
from embodiinfer.policies.activevln.processor_activevln import ActiveVLNProcessor
from embodiinfer.policies.activevln.prompt_activevln import render_turn_text


def prefix(**changes):
    return {
        "schema": EXECUTION_PREFIX_SCHEMA,
        "previous_step_id": 0,
        "remaining_actions": [[1, 12.5], [3, 15]],
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "unknown"},
        {"previous_step_id": -1},
        {"previous_step_id": True},
        {"remaining_actions": [[0, 0]]},
        {"remaining_actions": [[True, 25]]},
        {"remaining_actions": [[1, float("nan")]]},
        {"remaining_actions": [[2, 46]]},
        {"remaining_actions": [[1, 76]]},
        {"remaining_actions": [[1, 0]]},
        {"remaining_actions": [[1, True]]},
        {"remaining_actions": [[1, 1]] * 4},
        {"remaining_actions": [[1]]},
        {"unexpected": 0},
    ],
)
def test_invalid_prefix_is_rejected(changes):
    with pytest.raises(ValueError):
        ExecutionPrefix.parse(prefix(**changes), step_id=1)


def test_prefix_has_canonical_ack_and_previous_step_identity():
    value = ExecutionPrefix.parse(prefix(), step_id=1)
    assert value.to_wire()["remaining_actions"] == [[1, 12.5], [3, 15.0]]
    assert value.digest == ExecutionPrefix.parse(value.to_wire()).digest
    assert "12.500 cm" in value.prompt_context() and "15.000 degrees" in value.prompt_context()
    with pytest.raises(ValueError, match="previous step"):
        ExecutionPrefix.parse(prefix(), step_id=2)
    with pytest.raises(ValueError, match="previous step"):
        ExecutionPrefix.parse(prefix(), step_id=0)
    assert "have finished" in ExecutionPrefix.parse(prefix(remaining_actions=[])).prompt_context()


def processor():
    # No tokenizer download or checkpoint load; exercise the shared prompt path
    # used by CPU preprocessing and GPU image-layout caching.
    value = ActiveVLNProcessor.__new__(ActiveVLNProcessor)
    value.action_space = "r2r"
    value._processor = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: repr(messages))
    return value


@pytest.mark.parametrize("initial", [True, False])
def test_disabled_prefix_preserves_serialized_prompt_exactly(initial):
    value = processor()
    obs = SimpleNamespace(instruction="Go to the door.", metadata={})
    assert value.turn_text(obs, initial=initial) == render_turn_text(
        value._processor, obs.instruction, initial=initial
    )


def test_context_is_in_user_turn_and_initial_or_rxr_prefix_fails():
    value = processor()
    obs = SimpleNamespace(instruction="Go to the door.", metadata={"activevln_execution": prefix()})
    text = value.turn_text(obs, initial=False)
    assert text.count("<|im_start|>user") == 1 and "<|im_start|>system" not in text
    assert "continuation AFTER" in text and "12.500 cm" in text
    assert text.endswith("<|im_start|>assistant\n")
    with pytest.raises(ValueError, match="subsequent R2R"):
        value.turn_text(obs, initial=True)
    value.action_space = "rxr"
    with pytest.raises(ValueError, match="subsequent R2R"):
        value.turn_text(obs, initial=False)
