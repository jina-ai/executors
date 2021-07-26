__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='session', autouse=True)
def create_model_weights():
    path_to_model = os.path.join(TEST_DIR, 'model', 'model_state_dict.pth')
    if not os.path.isfile(path_to_model):
        os.system(f'python {os.path.join(TEST_DIR, "model", "external_model.py")}')

    yield

    if os.path.exists(path_to_model):
        os.remove(path_to_model)
