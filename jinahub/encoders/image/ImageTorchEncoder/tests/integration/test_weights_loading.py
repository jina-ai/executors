__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

from torch import hub
from pytest_mock import MockerFixture

from ...torch_encoder import ImageTorchEncoder



def test_load_from_url(tmpdir: str, mocker: MockerFixture) -> None:
    os.environ['TORCH_HOME'] = str(tmpdir)
    spy = mocker.spy(hub, 'urlopen')

    _ = ImageTorchEncoder(model_name='mobilenet_v2')

    assert os.path.isfile(os.path.join(tmpdir, 'hub', 'checkpoints', 'mobilenet_v2-b0353104.pth'))
    assert spy.call_count == 1
