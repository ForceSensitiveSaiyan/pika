"""Tests for query cache functionality."""

import time
import pytest
from unittest.mock import patch, MagicMock


class TestQueryCacheBasic:
    """Tests for basic query cache operations."""

    def test_cache_starts_empty(self):
        """Verify cache starts with no entries."""
        from pika.services.rag import QueryCache

        cache = QueryCache(max_size=100, ttl=300)
        assert cache.size() == 0

    def test_cache_set_and_get(self):
        """Verify items can be stored and retrieved."""
        from pika.services.rag import QueryCache, QueryResult, Confidence

        cache = QueryCache(max_size=100, ttl=300)

        result = QueryResult(
            answer="Test answer",
            sources=[],
            confidence=Confidence.HIGH,
        )

        cache.set("What is PIKA?", doc_count=5, chunk_count=100, result=result)

        retrieved = cache.get("What is PIKA?", doc_count=5, chunk_count=100)
        assert retrieved is not None
        assert retrieved.answer == "Test answer"
        assert retrieved.confidence == Confidence.HIGH

    def test_cache_miss_for_nonexistent_key(self):
        """Verify cache miss returns None."""
        from pika.services.rag import QueryCache

        cache = QueryCache(max_size=100, ttl=300)

        result = cache.get("Nonexistent question", doc_count=1, chunk_count=10)
        assert result is None

    def test_cache_key_includes_counts(self):
        """Verify cache key depends on document and chunk counts."""
        from pika.services.rag import QueryCache, QueryResult, Confidence

        cache = QueryCache(max_size=100, ttl=300)

        result = QueryResult(answer="Test", sources=[], confidence=Confidence.MEDIUM)

        # Store with specific counts
        cache.set("Question?", doc_count=5, chunk_count=100, result=result)

        # Same question, different counts = cache miss
        assert cache.get("Question?", doc_count=10, chunk_count=100) is None
        assert cache.get("Question?", doc_count=5, chunk_count=200) is None

        # Same question, same counts = cache hit
        assert cache.get("Question?", doc_count=5, chunk_count=100) is not None

    def test_cache_normalizes_question(self):
        """Verify cache normalizes question text for key generation."""
        from pika.services.rag import QueryCache, QueryResult, Confidence

        cache = QueryCache(max_size=100, ttl=300)

        result = QueryResult(answer="Test", sources=[], confidence=Confidence.MEDIUM)

        # Store with one format
        cache.set("What is PIKA?", doc_count=5, chunk_count=100, result=result)

        # Retrieve with different casing/whitespace
        assert cache.get("what is pika?", doc_count=5, chunk_count=100) is not None
        assert cache.get("  What is PIKA?  ", doc_count=5, chunk_count=100) is not None


class TestQueryCacheTTL:
    """Tests for cache TTL (time-to-live) expiration."""

    def test_cache_returns_fresh_entries(self):
        """Verify cache returns entries within TTL."""
        from pika.services.rag import QueryCache, QueryResult, Confidence

        cache = QueryCache(max_size=100, ttl=300)  # 5 minute TTL

        result = QueryResult(answer="Fresh", sources=[], confidence=Confidence.HIGH)
        cache.set("Question?", doc_count=1, chunk_count=10, result=result)

        # Immediately retrieve - should work
        retrieved = cache.get("Question?", doc_count=1, chunk_count=10)
        assert retrieved is not None
        assert retrieved.answer == "Fresh"

    def test_cache_expires_old_entries(self):
        """Verify cache expires entries past TTL."""
        from pika.services.rag import QueryCache, QueryResult, Confidence

        cache = QueryCache(max_size=100, ttl=1)  # 1 second TTL

        result = QueryResult(answer="Old", sources=[], confidence=Confidence.HIGH)
        cache.set("Question?", doc_count=1, chunk_count=10, result=result)

        # Wait for TTL to expire
        time.sleep(1.5)

        # Should be expired
        retrieved = cache.get("Question?", doc_count=1, chunk_count=10)
        assert retrieved is None

    def test_expired_entries_are_removed(self):
        """Verify expired entries are cleaned up on access."""
        from pika.services.rag import QueryCache, QueryResult, Confidence

        cache = QueryCache(max_size=100, ttl=1)

        result = QueryResult(answer="Test", sources=[], confidence=Confidence.HIGH)
        cache.set("Question?", doc_count=1, chunk_count=10, result=result)

        assert cache.size() == 1

        # Wait for TTL
        time.sleep(1.5)

        # Access the expired entry
        cache.get("Question?", doc_count=1, chunk_count=10)

        # Entry should be removed
        assert cache.size() == 0


class TestQueryCacheLRU:
    """Tests for LRU (Least Recently Used) eviction."""

    def test_cache_evicts_oldest_when_full(self):
        """Verify oldest entries are evicted when cache is full."""
        from pika.services.rag import QueryCache, QueryResult, Confidence

        cache = QueryCache(max_size=3, ttl=300)

        for i in range(3):
            result = QueryResult(answer=f"Answer {i}", sources=[], confidence=Confidence.MEDIUM)
            cache.set(f"Question {i}?", doc_count=1, chunk_count=10, result=result)

        assert cache.size() == 3

        # Add a 4th entry - should evict the oldest (Question 0)
        result = QueryResult(answer="Answer 3", sources=[], confidence=Confidence.MEDIUM)
        cache.set("Question 3?", doc_count=1, chunk_count=10, result=result)

        assert cache.size() == 3
        assert cache.get("Question 0?", doc_count=1, chunk_count=10) is None
        assert cache.get("Question 1?", doc_count=1, chunk_count=10) is not None
        assert cache.get("Question 2?", doc_count=1, chunk_count=10) is not None
        assert cache.get("Question 3?", doc_count=1, chunk_count=10) is not None

    def test_cache_access_updates_lru_order(self):
        """Verify accessing an entry moves it to most recent."""
        from pika.services.rag import QueryCache, QueryResult, Confidence

        cache = QueryCache(max_size=3, ttl=300)

        for i in range(3):
            result = QueryResult(answer=f"Answer {i}", sources=[], confidence=Confidence.MEDIUM)
            cache.set(f"Question {i}?", doc_count=1, chunk_count=10, result=result)

        # Access Question 0 to make it most recently used
        cache.get("Question 0?", doc_count=1, chunk_count=10)

        # Add new entry - should evict Question 1 (now oldest)
        result = QueryResult(answer="Answer 3", sources=[], confidence=Confidence.MEDIUM)
        cache.set("Question 3?", doc_count=1, chunk_count=10, result=result)

        assert cache.get("Question 0?", doc_count=1, chunk_count=10) is not None  # Was accessed
        assert cache.get("Question 1?", doc_count=1, chunk_count=10) is None  # Evicted
        assert cache.get("Question 2?", doc_count=1, chunk_count=10) is not None
        assert cache.get("Question 3?", doc_count=1, chunk_count=10) is not None


class TestQueryCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_clears_all_entries(self):
        """Verify invalidate() clears the entire cache."""
        from pika.services.rag import QueryCache, QueryResult, Confidence

        cache = QueryCache(max_size=100, ttl=300)

        for i in range(10):
            result = QueryResult(answer=f"Answer {i}", sources=[], confidence=Confidence.MEDIUM)
            cache.set(f"Question {i}?", doc_count=1, chunk_count=10, result=result)

        assert cache.size() == 10

        cache.invalidate()

        assert cache.size() == 0

        # All entries should be gone
        for i in range(10):
            assert cache.get(f"Question {i}?", doc_count=1, chunk_count=10) is None

    def test_invalidate_query_cache_function(self):
        """Verify invalidate_query_cache() function works."""
        import pika.services.rag as rag_module

        # Reset the global cache
        rag_module._query_cache = None

        from pika.services.rag import get_query_cache, invalidate_query_cache, QueryResult, Confidence

        cache = get_query_cache()
        result = QueryResult(answer="Test", sources=[], confidence=Confidence.HIGH)
        cache.set("Question?", doc_count=1, chunk_count=10, result=result)

        assert cache.size() > 0

        invalidate_query_cache()

        assert cache.size() == 0


class TestQueryCacheMetrics:
    """Tests for cache metrics integration."""

    def test_cache_hit_increments_counter(self):
        """Verify cache hit increments the hits counter."""
        from pika.services.rag import QueryCache, QueryResult, Confidence
        from pika.services.metrics import QUERY_CACHE_HITS

        initial_value = QUERY_CACHE_HITS._value.get()

        cache = QueryCache(max_size=100, ttl=300)
        result = QueryResult(answer="Test", sources=[], confidence=Confidence.HIGH)
        cache.set("Question?", doc_count=1, chunk_count=10, result=result)

        # Hit the cache
        cache.get("Question?", doc_count=1, chunk_count=10)

        new_value = QUERY_CACHE_HITS._value.get()
        assert new_value > initial_value

    def test_cache_miss_increments_counter(self):
        """Verify cache miss increments the misses counter."""
        from pika.services.rag import QueryCache
        from pika.services.metrics import QUERY_CACHE_MISSES

        initial_value = QUERY_CACHE_MISSES._value.get()

        cache = QueryCache(max_size=100, ttl=300)

        # Miss the cache
        cache.get("Nonexistent?", doc_count=1, chunk_count=10)

        new_value = QUERY_CACHE_MISSES._value.get()
        assert new_value > initial_value


class TestQueryCacheSingleton:
    """Tests for query cache singleton management."""

    def test_get_query_cache_returns_instance(self):
        """Verify get_query_cache returns a QueryCache instance."""
        import pika.services.rag as rag_module

        # Reset singleton
        rag_module._query_cache = None

        from pika.services.rag import get_query_cache, QueryCache

        cache = get_query_cache()
        assert isinstance(cache, QueryCache)

    def test_get_query_cache_returns_same_instance(self):
        """Verify get_query_cache returns singleton."""
        import pika.services.rag as rag_module

        # Reset singleton
        rag_module._query_cache = None

        from pika.services.rag import get_query_cache

        cache1 = get_query_cache()
        cache2 = get_query_cache()
        assert cache1 is cache2

    def test_cache_uses_config_values(self):
        """Verify cache uses settings from config."""
        import pika.services.rag as rag_module
        from pika.services.rag import QueryCache

        # Reset singleton
        rag_module._query_cache = None

        # Create a cache with custom values directly
        cache = QueryCache(max_size=50, ttl=600)
        assert cache.max_size == 50
        assert cache.ttl == 600

        # Verify the singleton uses config values by checking defaults
        from pika.config import get_settings
        settings = get_settings()
        rag_module._query_cache = None
        from pika.services.rag import get_query_cache

        cache = get_query_cache()
        assert cache.max_size == settings.query_cache_max_size
        assert cache.ttl == settings.query_cache_ttl


class TestQueryCacheThreadSafety:
    """Tests for cache thread safety."""

    def test_concurrent_reads_and_writes(self):
        """Verify cache handles concurrent operations safely."""
        from pika.services.rag import QueryCache, QueryResult, Confidence
        import threading

        cache = QueryCache(max_size=100, ttl=300)
        errors = []

        def writer():
            try:
                for i in range(100):
                    result = QueryResult(answer=f"Answer {i}", sources=[], confidence=Confidence.MEDIUM)
                    cache.set(f"Q{i}?", doc_count=1, chunk_count=10, result=result)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"Q{i}?", doc_count=1, chunk_count=10)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
