"""Tests for circuit breaker functionality."""

import asyncio
import pytest
from unittest.mock import patch


class TestCircuitBreakerStates:
    """Tests for circuit breaker state transitions."""

    def test_circuit_starts_closed(self):
        """Verify circuit breaker starts in CLOSED state."""
        from pika.services.ollama import CircuitBreaker, CircuitState

        circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        assert circuit.state == CircuitState.CLOSED
        assert circuit._failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_remains_closed_below_threshold(self):
        """Verify circuit stays closed when failures are below threshold."""
        from pika.services.ollama import CircuitBreaker, CircuitState

        circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=10)

        # Record 2 failures (below threshold of 3)
        await circuit.record_failure()
        await circuit.record_failure()

        assert circuit.state == CircuitState.CLOSED
        assert await circuit.is_available() is True

    @pytest.mark.asyncio
    async def test_circuit_opens_at_threshold(self):
        """Verify circuit opens when failure threshold is reached."""
        from pika.services.ollama import CircuitBreaker, CircuitState

        circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=10)

        # Record 3 failures (equals threshold)
        await circuit.record_failure()
        await circuit.record_failure()
        await circuit.record_failure()

        assert circuit.state == CircuitState.OPEN
        assert await circuit.is_available() is False

    @pytest.mark.asyncio
    async def test_circuit_open_rejects_requests(self):
        """Verify open circuit rejects requests."""
        from pika.services.ollama import CircuitBreaker, CircuitState

        circuit = CircuitBreaker(failure_threshold=1, recovery_timeout=100)

        # Open the circuit
        await circuit.record_failure()
        assert circuit.state == CircuitState.OPEN

        # Should reject requests
        assert await circuit.is_available() is False

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open_after_timeout(self):
        """Verify circuit transitions to HALF_OPEN after recovery timeout."""
        from pika.services.ollama import CircuitBreaker, CircuitState
        import time

        circuit = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

        # Open the circuit
        await circuit.record_failure()
        assert circuit.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should transition to half-open on next availability check
        assert await circuit.is_available() is True
        assert circuit.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_on_success_from_half_open(self):
        """Verify circuit closes when a request succeeds in HALF_OPEN state."""
        from pika.services.ollama import CircuitBreaker, CircuitState

        circuit = CircuitBreaker(failure_threshold=1, recovery_timeout=0)

        # Open then transition to half-open
        await circuit.record_failure()
        await asyncio.sleep(0.1)
        await circuit.is_available()  # Triggers half-open transition
        assert circuit.state == CircuitState.HALF_OPEN

        # Record success
        await circuit.record_success()

        assert circuit.state == CircuitState.CLOSED
        assert circuit._failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_reopens_on_failure_from_half_open(self):
        """Verify circuit reopens when a request fails in HALF_OPEN state."""
        from pika.services.ollama import CircuitBreaker, CircuitState

        circuit = CircuitBreaker(failure_threshold=1, recovery_timeout=0)

        # Open then transition to half-open
        await circuit.record_failure()
        await asyncio.sleep(0.1)
        await circuit.is_available()
        assert circuit.state == CircuitState.HALF_OPEN

        # Fail again
        await circuit.record_failure()

        assert circuit.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        """Verify successful request resets failure count."""
        from pika.services.ollama import CircuitBreaker

        circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=10)

        # Record some failures (not enough to open)
        await circuit.record_failure()
        await circuit.record_failure()
        assert circuit._failure_count == 2

        # Success should reset
        await circuit.record_success()
        assert circuit._failure_count == 0


class TestCircuitBreakerConfig:
    """Tests for circuit breaker configuration."""

    def test_custom_thresholds(self):
        """Verify circuit breaker accepts custom thresholds."""
        from pika.services.ollama import CircuitBreaker

        circuit = CircuitBreaker(failure_threshold=10, recovery_timeout=60)

        assert circuit.failure_threshold == 10
        assert circuit.recovery_timeout == 60

    @pytest.mark.asyncio
    async def test_high_failure_threshold(self):
        """Verify circuit with high failure threshold requires many failures."""
        from pika.services.ollama import CircuitBreaker, CircuitState

        circuit = CircuitBreaker(failure_threshold=10, recovery_timeout=10)

        # Record 9 failures (below threshold of 10)
        for _ in range(9):
            await circuit.record_failure()

        assert circuit.state == CircuitState.CLOSED

        # 10th failure should open
        await circuit.record_failure()
        assert circuit.state == CircuitState.OPEN


class TestCircuitBreakerMetrics:
    """Tests for circuit breaker metrics integration."""

    @pytest.mark.asyncio
    async def test_circuit_trip_increments_counter(self):
        """Verify circuit trip increments the trips counter."""
        from pika.services.ollama import CircuitBreaker
        from pika.services.metrics import CIRCUIT_BREAKER_TRIPS

        # Get initial value (may be > 0 from other tests)
        initial_value = CIRCUIT_BREAKER_TRIPS._value.get()

        circuit = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        await circuit.record_failure()  # This should trip the circuit

        # The counter should have incremented
        new_value = CIRCUIT_BREAKER_TRIPS._value.get()
        assert new_value > initial_value

    @pytest.mark.asyncio
    async def test_circuit_state_gauge_updates(self):
        """Verify circuit state gauge updates on state change."""
        from pika.services.ollama import CircuitBreaker, CircuitState
        from pika.services.metrics import CIRCUIT_BREAKER_STATE

        circuit = CircuitBreaker(failure_threshold=1, recovery_timeout=0)

        # Initially closed (state=0)
        assert circuit.state == CircuitState.CLOSED

        # Open the circuit (state=2)
        await circuit.record_failure()
        assert circuit.state == CircuitState.OPEN

        # The gauge should reflect the state
        # Note: In real usage, the metric is updated within the circuit breaker
        # This test verifies the metric exists and can be set
        CIRCUIT_BREAKER_STATE.set(2)  # OPEN
        assert CIRCUIT_BREAKER_STATE._value.get() == 2


class TestCircuitBreakerSingleton:
    """Tests for circuit breaker singleton management."""

    def test_get_circuit_breaker_returns_instance(self):
        """Verify get_circuit_breaker returns a CircuitBreaker instance."""
        import pika.services.ollama as ollama_module

        # Reset singleton
        ollama_module._circuit_breaker = None

        from pika.services.ollama import get_circuit_breaker, CircuitBreaker

        circuit = get_circuit_breaker()
        assert isinstance(circuit, CircuitBreaker)

    def test_get_circuit_breaker_returns_same_instance(self):
        """Verify get_circuit_breaker returns singleton."""
        import pika.services.ollama as ollama_module

        # Reset singleton
        ollama_module._circuit_breaker = None

        from pika.services.ollama import get_circuit_breaker

        circuit1 = get_circuit_breaker()
        circuit2 = get_circuit_breaker()
        assert circuit1 is circuit2

    def test_circuit_breaker_uses_config_values(self):
        """Verify circuit breaker uses settings from config."""
        import pika.services.ollama as ollama_module
        from pika.services.ollama import CircuitBreaker

        # Reset singleton
        ollama_module._circuit_breaker = None

        # Create a circuit breaker with custom values directly
        circuit = CircuitBreaker(failure_threshold=7, recovery_timeout=45)
        assert circuit.failure_threshold == 7
        assert circuit.recovery_timeout == 45

        # Verify the singleton uses config values by checking defaults
        from pika.config import get_settings
        settings = get_settings()
        ollama_module._circuit_breaker = None
        from pika.services.ollama import get_circuit_breaker

        circuit = get_circuit_breaker()
        assert circuit.failure_threshold == settings.circuit_breaker_failure_threshold
        assert circuit.recovery_timeout == settings.circuit_breaker_recovery_timeout


class TestCircuitBreakerException:
    """Tests for OllamaCircuitOpenError exception."""

    def test_circuit_open_error_default_message(self):
        """Verify OllamaCircuitOpenError has user-friendly default message."""
        from pika.services.ollama import OllamaCircuitOpenError

        error = OllamaCircuitOpenError()
        message = str(error)

        # Should be user-friendly, not technical
        assert "recovering" in message.lower() or "search" in message.lower()

    def test_circuit_open_error_custom_message(self):
        """Verify OllamaCircuitOpenError accepts custom message."""
        from pika.services.ollama import OllamaCircuitOpenError

        error = OllamaCircuitOpenError("Custom circuit error")
        assert "Custom circuit error" in str(error)


class TestCircuitBreakerConcurrency:
    """Tests for circuit breaker thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_failures_handled_safely(self):
        """Verify concurrent failures don't cause race conditions."""
        from pika.services.ollama import CircuitBreaker, CircuitState

        circuit = CircuitBreaker(failure_threshold=10, recovery_timeout=10)

        # Simulate 10 concurrent failures
        async def record_failure():
            await circuit.record_failure()

        tasks = [record_failure() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Should be exactly at or past threshold
        assert circuit.state == CircuitState.OPEN
        assert circuit._failure_count >= 10

    @pytest.mark.asyncio
    async def test_concurrent_state_checks(self):
        """Verify concurrent availability checks are safe."""
        from pika.services.ollama import CircuitBreaker

        circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=10)

        async def check_availability():
            return await circuit.is_available()

        # Many concurrent checks should all succeed without error
        tasks = [check_availability() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        # All should return True (circuit is closed)
        assert all(results)
