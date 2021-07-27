from jina import Document, DocumentArray
import pytest

from .. import RedisStorage
from pytest_mock_resources import create_redis_fixture

redis = create_redis_fixture()


@pytest.fixture(scope='function')
def redis_kwargs(redis):
    return redis.pmr_credentials.as_redis_kwargs()


@pytest.fixture(scope='function')
def indexer(redis_kwargs):
    return RedisStorage(hostname=redis_kwargs['host'], port=redis_kwargs['port'])


@pytest.fixture(scope='function')
def docs():
    return DocumentArray([
        Document(content=value)
        for value in ['cat', 'dog', 'crow', 'pikachu', 'magikarp']
    ])
