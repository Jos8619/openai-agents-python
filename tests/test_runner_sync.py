import pytest

from agents import Agent, AgentsException, Runner

from .fake_model import FakeModel
from .test_responses import get_text_message


def test_run_sync_uses_asyncio_run_when_no_loop():
    model = FakeModel()
    agent = Agent(
        name="test",
        model=model,
    )
    model.set_next_output([get_text_message("sync_result")])

    result = Runner.run_sync(agent, input="hello")

    assert result.final_output == "sync_result"


@pytest.mark.asyncio
async def test_run_sync_rejected_when_loop_running():
    model = FakeModel()
    agent = Agent(
        name="test",
        model=model,
    )

    with pytest.raises(AgentsException):
        Runner.run_sync(agent, input="hello")
