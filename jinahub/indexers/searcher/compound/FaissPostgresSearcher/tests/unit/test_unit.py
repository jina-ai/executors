import os
import time

import numpy as np
import pytest
from jina import Document, Flow

from ... import FaissPostgresSearcher


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


d_embedding = np.array([1, 1, 1, 1, 1, 1, 1])
c_embedding = np.array([2, 2, 2, 2, 2, 2, 2])

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.join(cur_dir, '..', 'docker-compose.yml')


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_shards_str(docker_compose):
    with Flow().load_config(
        '''
    jtype: Flow
    executors:
      - name: text_indexer
        shards: 1
        uses: 'FaissPostgresSearcher'
        uses_with:
          startup_sync_args: 
            only_delta: True
          total_shards: 1
    '''
    ) as f:
        f.search([Document() for _ in range(20)])
