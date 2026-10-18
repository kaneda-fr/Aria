"""The E2E run must fail on attempted external calls, not just successful ones."""
import unittest

from scripts.e2e.run import audit_network


class NetworkAuditTests(unittest.TestCase):
    def test_loopback(self):
        trace = 'connect(4, {sa_family=AF_INET, sin_port=htons(11435), sin_addr=inet_addr("127.0.0.1")}, 16) = 0'
        self.assertTrue(audit_network(trace)["passed"])

    def test_failed_external_attempt(self):
        trace = 'connect(4, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("203.0.113.1")}, 16) = -1 ENETUNREACH'
        self.assertFalse(audit_network(trace)["passed"])

    def test_ipv6_external_attempt(self):
        trace = 'connect(4, {sa_family=AF_INET6, sin6_port=htons(443), inet_pton(AF_INET6, "2001:db8::1", &sin6_addr)}, 28) = -1 ENETUNREACH'
        self.assertFalse(audit_network(trace)["passed"])

    def test_dns_attempt(self):
        trace = 'sendto(4, "query", 5, 0, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("127.0.0.53")}, 16) = 5'
        self.assertFalse(audit_network(trace)["passed"])

    def test_missing_evidence(self):
        self.assertFalse(audit_network("")["passed"])
