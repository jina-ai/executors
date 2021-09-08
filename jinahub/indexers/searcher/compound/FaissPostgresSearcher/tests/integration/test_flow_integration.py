import os
import time

import pytest
from jina import Document, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.join(cur_dir, '..', 'docker-compose.yml')


@pytest.fixture()
def docker_compose(request):
    os.system(
        f"docker-compose -f {request.param} --project-directory . up  --build -d "
        f"--remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker-compose -f {request.param} --project-directory . down "
        f"--remove-orphans"
    )


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_integration_parallel(docker_compose):
    # test issue reported by @florian
    SHARDS = 3

    with Flow().add(
        uses='FaissPostgresSearcher', shards=SHARDS, uses_with={'total_shards': 3}
    ) as f:
        f.index(Document())
