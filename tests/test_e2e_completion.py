"""A greeting's audio must not satisfy a subsequent request in the same session."""
import unittest
from unittest.mock import AsyncMock, patch

from scripts.e2e.run import wait_output


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_target_request_output(self):
        prefix = [
            {"event": "ARIA.LLM.Request", "user": "Bonjour"},
            {"event": "ARIA.LLM.Done"},
            {"event": "E2E.AudioFetched"},
            {"event": "ARIA.LLM.Request", "user": "Une histoire avec un chat"},
        ]
        before = {"events": prefix, "speech_busy": False}
        complete = {"events": prefix + [{"event": "ARIA.LLM.Done"},
                                       {"event": "E2E.AudioFetched"}], "speech_busy": False}

        class Response:
            def __init__(self, state):
                self.state = state

            def json(self):
                return self.state

        client = AsyncMock()
        client.get.side_effect = [Response(before), Response(complete)]
        with patch("scripts.e2e.run.asyncio.sleep", new_callable=AsyncMock):
            result = await wait_output(client, 0, keyword="chat")
        self.assertIs(result, complete)
        self.assertEqual(client.get.await_count, 2)
