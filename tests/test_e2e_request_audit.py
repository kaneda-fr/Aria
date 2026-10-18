import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('audit_requests', Path(__file__).resolve().parents[1] / 'scripts/e2e/audit_requests.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RequestAuditTests(unittest.TestCase):
    def event(self, index, status):
        return {'event': 'ARIA.Request.Status', 'index': index, 'status': status, 'request_id': 'one'}

    def test_dropped_accepted_request_fails(self):
        self.assertFalse(module.audit_requests({'cases': [{'events': [self.event(1, 'accepted')]}]})['passed'])

    def test_overlap_does_not_double_count_terminal(self):
        events = [self.event(1, 'accepted'), self.event(2, 'completed')]
        result = module.audit_requests({'cases': [{'events': events}, {'events': events}]})
        self.assertTrue(result['passed'])
        self.assertEqual(result['terminal'], 1)

    def test_duplicate_terminal_fails(self):
        events = [self.event(1, 'accepted'), self.event(2, 'completed'), self.event(3, 'completed')]
        self.assertFalse(module.audit_requests({'cases': [{'events': events}]})['passed'])
