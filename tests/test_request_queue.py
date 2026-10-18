import asyncio
import contextlib
import unittest
from app.llm.request_queue import RequestQueue


def submit(queue, text='hello', source='voice', **kwargs):
    return queue.submit(text=text, session_id=source, speaker_user='unknown',
                        source=source, speak_response=True, use_tools=True, **kwargs)


class RequestQueueTests(unittest.IsolatedAsyncioTestCase):
    async def stop(self, worker):
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    async def test_mixed_transport_overload_preserves_fifo_and_one_worker(self):
        queue = RequestQueue()
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []
        async def process(job):
            seen.append(job.text)
            entered.set()
            await release.wait()
            return {'success': True}
        worker = asyncio.create_task(queue.run(process))
        try:
            first = submit(queue, 'first', 'api')
            await entered.wait()
            second = submit(queue, 'second', 'voice')
            rejected = submit(queue, 'third', 'api')
            self.assertEqual((await rejected.result)['error'], 'busy')
            self.assertFalse(second.result.done())
            release.set()
            await asyncio.wait_for(asyncio.gather(first.result, second.result), 1)
            self.assertEqual(seen, ['first', 'second'])
        finally:
            await self.stop(worker)

    async def test_expired_job_never_executes_and_notifies(self):
        queue, seen, statuses = RequestQueue(), [], []
        async def notify(event): statuses.append(event)
        job = submit(queue, notify=notify)
        job.queued_at -= 100
        async def process(job): seen.append(job.text)
        worker = asyncio.create_task(queue.run(process))
        try:
            self.assertEqual((await asyncio.wait_for(job.result, 1))['status'], 'expired')
            await asyncio.sleep(0)
            self.assertEqual(seen, [])
            self.assertEqual(statuses[0]['error'], 'queue_expired')
        finally:
            await self.stop(worker)

    async def test_failed_request_does_not_kill_worker(self):
        queue = RequestQueue()
        async def process(job):
            if job.text == 'bad': raise RuntimeError('model unavailable')
            return {'success': True, 'response': 'ok'}
        worker = asyncio.create_task(queue.run(process))
        try:
            self.assertEqual((await submit(queue, 'bad').result)['status'], 'failed')
            self.assertEqual((await submit(queue, 'good').result)['response'], 'ok')
        finally:
            await self.stop(worker)

    async def test_cancelled_pending_request_never_executes(self):
        queue, seen = RequestQueue(), []
        job = submit(queue)
        queue.cancel_pending(job)
        async def process(job): seen.append(job.text)
        worker = asyncio.create_task(queue.run(process))
        try:
            await asyncio.wait_for(queue.pending.join(), 1)
            self.assertEqual(seen, [])
            self.assertEqual(job.result.result()['error'], 'caller_cancelled')
        finally:
            await self.stop(worker)

    async def test_shutdown_resolves_running_and_pending_and_rejects_new(self):
        queue, entered = RequestQueue(), asyncio.Event()
        async def process(job):
            entered.set()
            await asyncio.Event().wait()
        worker = asyncio.create_task(queue.run(process))
        first = submit(queue)
        await entered.wait()
        queue.cancel_pending(first)  # A disconnected caller must not retry/cancel a running action.
        self.assertFalse(first.result.done())
        second = submit(queue)
        await self.stop(worker)
        self.assertEqual(first.result.result()['error'], 'execution_interrupted')
        self.assertEqual(second.result.result()['error'], 'shutdown')
        self.assertEqual(submit(queue).result.result()['error'], 'shutdown')

    def test_no_unbounded_queue_configuration(self):
        for capacity in (0, -1, 17):
            with self.assertRaises(ValueError): RequestQueue(capacity=capacity)
