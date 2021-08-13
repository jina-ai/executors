# LightGBMRanker

LightGBMRanker learning to rank

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://LightGBMRanker')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://LightGBMRanker')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
