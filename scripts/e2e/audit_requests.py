"""Audit accepted-request accounting in an E2E report, independently of answer quality."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


def audit_requests(report: dict) -> dict:
    # Cases can contain overlapping event windows; server event indices are unique.
    events = {event['index']: event for case in report.get('cases', [])
              for event in case.get('events', []) if event['event'] == 'ARIA.Request.Status'}
    accepted, terminal = Counter(), Counter()
    outcomes = Counter()
    for event in events.values():
        status, request_id = event['status'], event['request_id']
        outcomes[status] += 1
        if status == 'accepted':
            accepted[request_id] += 1
        if status in {'completed', 'failed', 'expired', 'cancelled'}:
            terminal[request_id] += 1
    invalid = sorted(request_id for request_id in accepted.keys() | terminal.keys()
                     if accepted[request_id] != 1 or terminal[request_id] != 1)
    return {'passed': bool(accepted) and not invalid, 'accepted': sum(accepted.values()),
            'terminal': sum(terminal.values()), 'outcomes': dict(outcomes),
            'invalid_request_ids': invalid}


if __name__ == '__main__':
    result = audit_requests(json.loads(Path(sys.argv[1]).read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
