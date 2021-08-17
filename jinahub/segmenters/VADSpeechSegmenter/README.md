# VADSpeechSegmenter

VADSpeechSegmenter segments speech audio using Silero VAD (voice activity detector)


**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#-example)
- [🔍️ Reference](#-reference)


## 🌱 Prerequisites

> These are only needed if you download the source code and directly use the class. Not needed if you use the Jina Hub method below.

In case you want to install the dependencies locally run
```
pip install -r requirements.txt
```

## Usage

#### via Docker image (recommended)

```python
from jina import Flow

f = Flow().add(uses='jinahub+docker://VADSpeechSegmenter')
```

#### via source code

```python
from jina import Flow

f = Flow().add(uses='jinahub://VADSpeechSegmenter')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`


## Example

Work in progress.


## 🔍️ Reference
- [silero-vad](https://github.com/snakers4/silero-vad)
