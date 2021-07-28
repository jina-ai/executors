import os
import time

import pytest


@pytest.fixture(scope='function', autouse=True)
def patched_random_port(mocker):
    print('using random port fixture...')
    used_ports = set()
    from jina.helper import random_port

    def _random_port():

        for i in range(10):
            _port = random_port()

            if _port is not None and _port not in used_ports:
                used_ports.add(_port)
                return _port
        raise Exception('no available port')

    mocker.patch('jina.helper.random_port', new_callable=lambda: _random_port)


@pytest.fixture()
def docker_compose(request):
    os.system(
        f"docker-compose -f {request.param} --project-directory . up  --build -d --remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker-compose -f {request.param} --project-directory . down --remove-orphans"
    )
