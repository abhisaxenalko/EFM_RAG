from __future__ import annotations

import pytest

from app.llm import FakeLLM, LLMClient
from app.models.schemas import GradeOutput, Usage


@pytest.mark.asyncio
async def test_fake_llm_returns_schema_responses_in_order_and_records_prompts() -> None:
    first = GradeOutput(score="relevant", missing_aspect="")
    second = GradeOutput(score="ambiguous", missing_aspect="operating pressure")
    supplied_usage = Usage(
        model="scripted-model",
        input_tokens=12,
        output_tokens=3,
        cost_usd=0.001,
        latency_ms=8.5,
    )
    llm: LLMClient = FakeLLM(
        {GradeOutput: [first, (second, supplied_usage)]},
    )

    result_one, usage_one = await llm.complete_structured(
        system="Grade evidence.",
        user="The manual confirms the fault code.",
        schema=GradeOutput,
    )
    result_two, usage_two = await llm.complete_structured(
        system="Grade evidence.",
        user="The manual omits the operating pressure.",
        schema=GradeOutput,
    )

    assert result_one == first
    assert usage_one.model == "fake"
    assert result_two == second
    assert usage_two == supplied_usage
    assert isinstance(llm, FakeLLM)
    assert llm.calls == [
        ("Grade evidence.", "The manual confirms the fault code."),
        ("Grade evidence.", "The manual omits the operating pressure."),
    ]
    with pytest.raises(RuntimeError, match="structured schema GradeOutput"):
        await llm.complete_structured(
            system="Grade evidence.",
            user="No scripted response remains.",
            schema=GradeOutput,
        )
    assert llm.calls[-1] == ("Grade evidence.", "No scripted response remains.")


@pytest.mark.asyncio
async def test_fake_llm_text_queue_and_exhaustion_error() -> None:
    llm = FakeLLM(text_responses=["Check the valve."])
    text, usage = await llm.complete_text(system="Be concise.", user="What next?")
    assert text == "Check the valve."
    assert usage.model == "fake"
    assert llm.calls == [("Be concise.", "What next?")]

    with pytest.raises(RuntimeError, match="script exhausted for text completion"):
        await llm.complete_text(system="Be concise.", user="Anything else?")
    assert llm.calls[-1] == ("Be concise.", "Anything else?")
