import asyncio

import pytest

from agents import Agent
from agents._run_impl import QueueCompleteSentinel
from agents.result import RunResultStreaming


@pytest.mark.asyncio
async def test_stream_events_cleans_up_on_close():
    agent = Agent(name="test")
    result = RunResultStreaming(
        input="hi",
        new_items=[],
        raw_responses=[],
        final_output=None,
        input_guardrail_results=[],
        output_guardrail_results=[],
        current_agent=agent,
        current_turn=0,
        max_turns=1,
        _current_agent_output_schema=None,
        _trace=None,
    )

    block_event = asyncio.Event()
    result._run_impl_task = asyncio.create_task(block_event.wait())
    result._input_guardrails_task = asyncio.create_task(block_event.wait())
    result._output_guardrails_task = asyncio.create_task(block_event.wait())

    tasks = (
        result._run_impl_task,
        result._input_guardrails_task,
        result._output_guardrails_task,
    )

    stream = result.stream_events()
    await stream.aclose()

    assert all(task.cancelled() for task in tasks)
    assert result._run_impl_task is None
    assert result._input_guardrails_task is None
    assert result._output_guardrails_task is None


@pytest.mark.asyncio
async def test_stream_events_raises_stored_exception_after_cleanup():
    agent = Agent(name="test")
    result = RunResultStreaming(
        input="hi",
        new_items=[],
        raw_responses=[],
        final_output=None,
        input_guardrail_results=[],
        output_guardrail_results=[],
        current_agent=agent,
        current_turn=0,
        max_turns=1,
        _current_agent_output_schema=None,
        _trace=None,
    )

    await result._event_queue.put(QueueCompleteSentinel())
    result.is_complete = True
    result._stored_exception = RuntimeError("boom")

    stream = result.stream_events()

    with pytest.raises(RuntimeError):
        async for _ in stream:
            pass


@pytest.mark.asyncio
async def test_run_result_streaming_aclose_without_stream():
    agent = Agent(name="test")
    result = RunResultStreaming(
        input="hi",
        new_items=[],
        raw_responses=[],
        final_output=None,
        input_guardrail_results=[],
        output_guardrail_results=[],
        current_agent=agent,
        current_turn=0,
        max_turns=1,
        _current_agent_output_schema=None,
        _trace=None,
    )

    block_event = asyncio.Event()
    result._run_impl_task = asyncio.create_task(block_event.wait())

    await result.aclose()

    assert result._run_impl_task is None
