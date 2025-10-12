import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from src.data.collectors.cache import ResponseCache, CachedTank01Client


class TestResponseCache:
    """Tests for ResponseCache"""

    @pytest.fixture
    def temp_cache(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        cache = ResponseCache(cache_dir=temp_dir, ttl_days=30)
        yield cache
        shutil.rmtree(temp_dir)

    def test_cache_miss(self, temp_cache):
        """Test cache miss returns None"""
        result = temp_cache.get('test_endpoint', {'param1': 'value1'})
        assert result is None

    def test_cache_hit(self, temp_cache):
        """Test cache hit returns cached data"""
        endpoint = 'test_endpoint'
        params = {'param1': 'value1'}
        response = {'statusCode': 200, 'body': {'data': 'test'}}

        temp_cache.set(endpoint, params, response)

        cached_response = temp_cache.get(endpoint, params)

        assert cached_response is not None
        assert cached_response == response

    def test_cache_key_generation(self, temp_cache):
        """Test cache key is consistent for same params"""
        params1 = {'a': 1, 'b': 2}
        params2 = {'b': 2, 'a': 1}

        key1 = temp_cache._generate_cache_key('endpoint', params1)
        key2 = temp_cache._generate_cache_key('endpoint', params2)

        assert key1 == key2

    def test_cache_different_params(self, temp_cache):
        """Test different params create different cache entries"""
        endpoint = 'test_endpoint'
        response1 = {'data': 'response1'}
        response2 = {'data': 'response2'}

        temp_cache.set(endpoint, {'param': 'value1'}, response1)
        temp_cache.set(endpoint, {'param': 'value2'}, response2)

        cached1 = temp_cache.get(endpoint, {'param': 'value1'})
        cached2 = temp_cache.get(endpoint, {'param': 'value2'})

        assert cached1 == response1
        assert cached2 == response2

    def test_clear_cache(self, temp_cache):
        """Test clearing all cache"""
        temp_cache.set('endpoint1', {'p': 1}, {'data': 1})
        temp_cache.set('endpoint2', {'p': 2}, {'data': 2})

        count = temp_cache.clear()

        assert count == 2
        assert temp_cache.get('endpoint1', {'p': 1}) is None
        assert temp_cache.get('endpoint2', {'p': 2}) is None

    def test_cache_expiration(self, temp_cache):
        """Test cache expiration with TTL"""
        cache_short_ttl = ResponseCache(cache_dir=temp_cache.cache_dir, ttl_days=0)

        cache_short_ttl.set('endpoint', {'p': 1}, {'data': 'test'})

        result = cache_short_ttl.get('endpoint', {'p': 1})

        assert result is None


class TestCachedTank01Client:
    """Tests for CachedTank01Client"""

    @pytest.fixture
    def mock_client(self):
        """Create mock Tank01Client"""
        client = Mock()
        client.request_count = 0
        client.rate_limit = 1000
        client.get_daily_schedule = Mock(return_value={'statusCode': 200, 'body': []})
        client.get_betting_odds = Mock(return_value={'statusCode': 200, 'body': []})
        return client

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_cache_miss_calls_client(self, mock_client, temp_cache_dir):
        """Test cache miss calls underlying client"""
        cached_client = CachedTank01Client(mock_client, cache_dir=temp_cache_dir)

        result = cached_client.get_daily_schedule('20240101')

        assert mock_client.get_daily_schedule.called
        assert result == {'statusCode': 200, 'body': []}

    def test_cache_hit_does_not_call_client(self, mock_client, temp_cache_dir):
        """Test cache hit does not call underlying client"""
        cached_client = CachedTank01Client(mock_client, cache_dir=temp_cache_dir)

        cached_client.get_daily_schedule('20240101')
        mock_client.get_daily_schedule.reset_mock()

        cached_client.get_daily_schedule('20240101')

        assert not mock_client.get_daily_schedule.called

    def test_cache_stats(self, mock_client, temp_cache_dir):
        """Test cache statistics tracking"""
        cached_client = CachedTank01Client(mock_client, cache_dir=temp_cache_dir)

        cached_client.get_daily_schedule('20240101')
        cached_client.get_daily_schedule('20240101')
        cached_client.get_daily_schedule('20240102')

        stats = cached_client.get_cache_stats()

        assert stats['hits'] == 1
        assert stats['misses'] == 2
        assert stats['total'] == 3
        assert stats['hit_rate'] == 33.33

    def test_multiple_methods_cached(self, mock_client, temp_cache_dir):
        """Test multiple client methods are cached"""
        cached_client = CachedTank01Client(mock_client, cache_dir=temp_cache_dir)

        cached_client.get_daily_schedule('20240101')
        cached_client.get_betting_odds(game_date='20240101')

        assert mock_client.get_daily_schedule.call_count == 1
        assert mock_client.get_betting_odds.call_count == 1

        cached_client.get_daily_schedule('20240101')
        cached_client.get_betting_odds(game_date='20240101')

        assert mock_client.get_daily_schedule.call_count == 1
        assert mock_client.get_betting_odds.call_count == 1

    def test_get_request_count(self, mock_client, temp_cache_dir):
        """Test getting request count from underlying client"""
        mock_client.request_count = 42
        cached_client = CachedTank01Client(mock_client, cache_dir=temp_cache_dir)

        assert cached_client.get_request_count() == 42

    def test_get_remaining_requests(self, mock_client, temp_cache_dir):
        """Test getting remaining requests"""
        mock_client.request_count = 100
        mock_client.rate_limit = 1000
        cached_client = CachedTank01Client(mock_client, cache_dir=temp_cache_dir)

        assert cached_client.get_remaining_requests() == 900
