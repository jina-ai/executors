# CustomImageTorchEncoder

**CustomImageTorchEncoder** is a class that uses any custom pretrained model provided to extract embeddings for `Documents` containing images as `blob`.
It relies on having a [`state_dict`](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict) stored
together with a `python` file and `class` name to load the model from.



## Usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # here you can run your training logic
    path = 'model_state_dict.pth'
    model = CustomModel()
    torch.save(model.state_dict(), path)
```

```bash
python model.py
```


```python
import numpy as np
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://CustomImageTorchEncoder', 
                uses_with={'model_state_dict_path': 'model_state_dict.pth',
                               'layer_name': 'conv1',
                               'model_definition_file': 'model.py',
                               'model_class_name': 'CustomModel'},
                volumes='.:/workspace')
with f:
    resp = f.post(on='foo', inputs=Document(blob=np.random.rand(3, 224, 224)), return_results=True)
    assert resp[0].docs[0].embedding.shape == (10, )
```


## Reference
- https://pytorch.org/tutorials/beginner/saving_loading_models.html

<!-- version=v0.3 -->
