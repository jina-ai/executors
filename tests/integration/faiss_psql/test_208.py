import os

import pytest
from executor.faisspsql import FaissPostgresIndexer
from jina import Document, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.join(cur_dir, 'docker-compose.yml')

# fixes issue #208 https://github.com/jina-ai/executors/issues/208
@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_shards_str(docker_compose, tmp_path):
    with Flow().load_config(
        f"""
    jtype: Flow
    executors:
      - name: text_indexer
        shards: 1
        uses: FaissPostgresIndexer
        workspace: {tmp_path}
        uses_with:
          startup_sync_args: 
            only_delta: True
          total_shards: 1
    """
    ) as f:
        f.search([Document() for _ in range(20)])
