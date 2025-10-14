import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResponseCache:
    """File-based cache for API responses"""

    def __init__(self, cache_dir: str = 'data/cache', ttl_days: int = 30):
        """
        Initialize response cache.

        Args:
            cache_dir: Directory to store cached responses
            ttl_days: Time-to-live for cached responses in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days

    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Generate cache key from endpoint and parameters.

        Args:
            endpoint: API endpoint name
            params: Request parameters

        Returns:
            Cache key string
        """
        param_str = json.dumps(params, sort_keys=True)
        hash_obj = hashlib.md5(f"{endpoint}_{param_str}".encode())
        return hash_obj.hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get file path for cache key.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.json"

    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response if available and not expired.

        Args:
            endpoint: API endpoint name
            params: Request parameters

        Returns:
            Cached response or None if cache miss
        """
        cache_key = self._generate_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {endpoint} with {params}")
            return None

        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)

            cached_at = datetime.fromisoformat(cached_data['cached_at'])
            expiry = cached_at + timedelta(days=self.ttl_days)

            if datetime.now() > expiry:
                logger.debug(f"Cache expired: {endpoint} with {params}")
                cache_path.unlink()
                return None

            logger.debug(f"Cache hit: {endpoint} with {params}")
            return cached_data['response']

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid cache file {cache_path}: {e}")
            cache_path.unlink()
            return None

    def set(self, endpoint: str, params: Dict[str, Any], response: Dict[str, Any]) -> None:
        """
        Cache API response.

        Args:
            endpoint: API endpoint name
            params: Request parameters
            response: API response to cache
        """
        cache_key = self._generate_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        cached_data = {
            'endpoint': endpoint,
            'params': params,
            'response': response,
            'cached_at': datetime.now().isoformat()
        }

        with open(cache_path, 'w') as f:
            json.dump(cached_data, f)

        logger.debug(f"Cached response: {endpoint} with {params}")

    def clear(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob('*.json'):
            cache_file.unlink()
            count += 1
        logger.info(f"Cleared {count} cached responses")
        return count

    def clear_expired(self) -> int:
        """
        Clear only expired cached responses.

        Returns:
            Number of expired files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob('*.json'):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)

                cached_at = datetime.fromisoformat(cached_data['cached_at'])
                expiry = cached_at + timedelta(days=self.ttl_days)

                if datetime.now() > expiry:
                    cache_file.unlink()
                    count += 1

            except (json.JSONDecodeError, KeyError, ValueError):
                cache_file.unlink()
                count += 1

        logger.info(f"Cleared {count} expired cached responses")
        return count


class CachedTank01Client:
    """Wrapper around Tank01Client with response caching"""

    def __init__(self, client, cache_dir: str = 'data/cache', ttl_days: int = 30):
        """
        Initialize cached client.

        Args:
            client: Tank01Client instance to wrap
            cache_dir: Directory to store cached responses
            ttl_days: Time-to-live for cached responses in days
        """
        self.client = client
        self.cache = ResponseCache(cache_dir, ttl_days)
        self.cache_hits = 0
        self.cache_misses = 0

    def _cached_request(self, endpoint: str, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Make request with caching.

        Args:
            endpoint: API endpoint name for cache key
            method_name: Client method name to call
            *args: Positional arguments for client method
            **kwargs: Keyword arguments for client method

        Returns:
            API response (from cache or fresh)
        """
        cache_params = {'args': args, 'kwargs': kwargs}
        cached_response = self.cache.get(endpoint, cache_params)

        if cached_response is not None:
            self.cache_hits += 1
            return cached_response

        self.cache_misses += 1
        method = getattr(self.client, method_name)
        response = method(*args, **kwargs)
        self.cache.set(endpoint, cache_params, response)
        return response

    def get_daily_schedule(self, game_date: str) -> Dict[str, Any]:
        """Get NBA schedule with caching"""
        return self._cached_request('daily_schedule', 'get_daily_schedule', game_date)

    def get_betting_odds(self, **kwargs) -> Dict[str, Any]:
        """Get betting odds with caching"""
        return self._cached_request('betting_odds', 'get_betting_odds', **kwargs)

    def get_box_score(self, game_id: str, **kwargs) -> Dict[str, Any]:
        """Get box score with caching"""
        return self._cached_request('box_score', 'get_box_score', game_id, **kwargs)

    def get_dfs_salaries(self, **kwargs) -> Dict[str, Any]:
        """Get DFS salaries with caching"""
        return self._cached_request('dfs_salaries', 'get_dfs_salaries', **kwargs)

    def get_projections(self, **kwargs) -> Dict[str, Any]:
        """Get projections with caching"""
        return self._cached_request('projections', 'get_projections', **kwargs)

    def get_injuries(self, **kwargs) -> Dict[str, Any]:
        """Get injuries with caching"""
        return self._cached_request('injuries', 'get_injuries', **kwargs)

    def get_teams(self, **kwargs) -> Dict[str, Any]:
        """Get teams with caching"""
        return self._cached_request('teams', 'get_teams', **kwargs)

    def get_request_count(self) -> int:
        """Get client request count"""
        return self.client.request_count

    def get_remaining_requests(self) -> int:
        """Get remaining API requests"""
        return self.client.rate_limit - self.client.request_count

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hit/miss counts and hit rate
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0

        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total': total,
            'hit_rate': round(hit_rate, 2)
        }

    def clear_cache(self) -> int:
        """Clear all cached responses"""
        return self.cache.clear()

    def clear_expired_cache(self) -> int:
        """Clear expired cached responses"""
        return self.cache.clear_expired()
