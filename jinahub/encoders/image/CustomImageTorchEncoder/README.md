# ✨ CustomImageTorchEncoder

**CustomImageTorchEncoder** is a class that uses any custom pretrained model provided to extract embeddings for `Documents` containing images as `blob`.
It relies on having a [`state_dict`](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict) stored
together with a `python` file and `class` name to load the model from.

**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)
- 

## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

In case you want to install the dependencies locally run 
```
pip install -r requirements.txt
```

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://CustomImageTorchEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://CustomImageTorchEncoder'
```

#### using source code
Use the source code from JinaHub in your Python code:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://CustomImageTorchEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://CustomImageTorchEncoder'
```


## 🎉️ Example

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

### Inputs 

`Documents` must have `blob` content as images. There is no explicit requirement in the shape of the `arrays` contained by the `Documents`.
The input must be the one accepted by the model offered as pretrained. User may need to pay attention to the required preprocessing needed
at inference time for the images.

### Returns

`Documents` with `embedding` field


## 🔍️ Reference
- https://pytorch.org/tutorials/beginner/saving_loading_models.html
