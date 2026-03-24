"""Mock validator for testing miners locally.

Simulates validator queries using the subnet's protocol spec so you can
test a miner without deploying to the real network.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from bt_miner_skills.config.subnet_config import ProtocolSpec


@dataclass
class QueryResult:
    """Result of a single mock validator query."""

    request: dict[str, Any]
    response: dict[str, Any] | None
    latency_ms: float
    success: bool
    error: str | None = None


@dataclass
class TestReport:
    """Summary of a mock validation test run."""

    total_queries: int = 0
    successful: int = 0
    failed: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    results: list[QueryResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful / max(self.total_queries, 1)


def generate_sample_request(protocol: ProtocolSpec) -> dict[str, Any]:
    """Generate a sample request based on the protocol spec.

    Creates plausible test data for each field type.
    """
    type_samples: dict[str, Any] = {
        "str": "Hello, this is a test prompt. Please respond thoughtfully.",
        "int": 42,
        "float": 0.5,
        "bool": True,
        "list[str]": ["user", "assistant"],
        "list[int]": [1, 2, 3],
        "dict": {"key": "value"},
        "Optional[str]": "optional test value",
    }

    request = {}
    for field_name, field_type in protocol.request_fields.items():
        # Try exact match, then partial match
        if field_type in type_samples:
            request[field_name] = type_samples[field_type]
        elif "list" in field_type.lower():
            request[field_name] = type_samples.get("list[str]", ["test"])
        elif "str" in field_type.lower():
            request[field_name] = type_samples["str"]
        elif "int" in field_type.lower():
            request[field_name] = type_samples["int"]
        elif "float" in field_type.lower():
            request[field_name] = type_samples["float"]
        else:
            request[field_name] = f"<sample {field_type}>"

    return request


def validate_response(
    response: dict[str, Any], protocol: ProtocolSpec
) -> tuple[bool, list[str]]:
    """Check if a response matches the expected protocol format."""
    errors = []

    for field_name, field_type in protocol.response_fields.items():
        if field_name not in response:
            errors.append(f"Missing response field: {field_name}")
            continue
        value = response[field_name]
        if value is None:
            errors.append(f"Response field '{field_name}' is None")
        elif isinstance(value, str) and not value.strip():
            errors.append(f"Response field '{field_name}' is empty")

    return len(errors) == 0, errors


class MockValidator:
    """Simulates a Bittensor validator for local miner testing.

    Usage:
        validator = MockValidator(protocol_spec)
        report = await validator.run_test(forward_fn, num_queries=10)
        print(f"Success rate: {report.success_rate:.1%}")
    """

    def __init__(self, protocol: ProtocolSpec, timeout: float | None = None):
        self.protocol = protocol
        self.timeout = timeout or protocol.timeout_seconds

    async def query_miner(
        self,
        forward_fn,
        request: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Send a single query to the miner's forward function."""
        if request is None:
            request = generate_sample_request(self.protocol)

        start = time.monotonic()
        try:
            if asyncio.iscoroutinefunction(forward_fn):
                response = await asyncio.wait_for(
                    forward_fn(request), timeout=self.timeout
                )
            else:
                response = forward_fn(request)

            latency = (time.monotonic() - start) * 1000

            if isinstance(response, dict):
                resp_dict = response
            elif hasattr(response, "__dict__"):
                resp_dict = {
                    k: v
                    for k, v in response.__dict__.items()
                    if not k.startswith("_")
                }
            else:
                resp_dict = {"result": response}

            valid, errors = validate_response(resp_dict, self.protocol)

            return QueryResult(
                request=request,
                response=resp_dict,
                latency_ms=latency,
                success=valid,
                error="; ".join(errors) if errors else None,
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return QueryResult(
                request=request,
                response=None,
                latency_ms=latency,
                success=False,
                error=str(e),
            )

    async def run_test(
        self,
        forward_fn,
        num_queries: int = 10,
        custom_requests: list[dict[str, Any]] | None = None,
    ) -> TestReport:
        """Run a full test suite against the miner."""
        results = []

        requests = custom_requests or [
            generate_sample_request(self.protocol) for _ in range(num_queries)
        ]

        for req in requests:
            result = await self.query_miner(forward_fn, req)
            results.append(result)

        latencies = [r.latency_ms for r in results]
        successful = [r for r in results if r.success]

        return TestReport(
            total_queries=len(results),
            successful=len(successful),
            failed=len(results) - len(successful),
            avg_latency_ms=sum(latencies) / max(len(latencies), 1),
            max_latency_ms=max(latencies, default=0),
            min_latency_ms=min(latencies, default=0),
            results=results,
            errors=[r.error for r in results if r.error],
        )
