# DPRReaderRanker

**DPRReaderRanker** uses the [DPR Reader model](https://huggingface.co/transformers/model_doc/dpr.html) to perform two tasks

- re-rank the matches based on the query document
- extract answer spans from each match


The **DPR** model was originally proposed in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906). This ranker should be used together with **[DPRTextEncoder](https://hub.jina.ai/executor/awl0jxog)** to use the entire
DPR pipeline for QA tasks.

## Usage

Here's a simple example of usage for this executor

```python
from jina import Document, Flow
from jina.types.request import Response


question = Document(text='When was Napoleon born?')
matches = [
    Document(
        text='Napoléon Bonaparte[a] (born Napoleone di Buonaparte; 15 August 1769 – 5'
        ' May 1821), usually referred to as simply Napoleon in English'
    ),
    Document(
        text='On 20 March 1811, Marie Louise gave birth to a baby boy, whom Napoleon'
        ' made heir apparent and bestowed the title of King of Rome.'
    ),
]
question.matches.extend(matches)

f = Flow().add(
    uses='jinahub+docker://DPRReaderRanker', uses_with={'num_spans_per_match': 1}
)


def print_matches(response: Response):
    for match in response.data.docs[0].matches:
        rel_score = match.scores['relevance_score'].value
        span_score = match.scores['span_score'].value
        print(
            f'Napoleon was born on {match.text} [rel. score: {rel_score:.2%},'
            f' span score: {span_score:.2%}]'
        )


with f:
    f.post('/rank', question, on_done=print_matches)
```

```console
Napoleon was born on 15 august 1769 [rel. score: 57.56%, span score: 65.04%]
Napoleon was born on 20 march 1811 [rel. score: 7.13%, span score: 99.99%]
```

Note that this way the Executor will download the model every time it starts up. You can
re-use the cached model files by mounting the cache directory that the model is using
into the container. To do this, modify the Flow definition like this

```python
f = Flow().add(
    uses='jinahub+docker://DPRReaderRanker',
    volumes='/your/home/dir/.cache/huggingface:/root/.cache/huggingface'
)
```

### With GPU

This encoder also offers a GPU version under the `gpu` tag. To use it, make sure to pass `device='cuda'`, as the initialization parameter, and `gpus='all'` when adding the containerized Executor to the Flow. See the [Executor on GPU](https://docs.jina.ai/tutorials/gpu-executor/) section of Jina documentation for more details.

Here's how you would modify the example above to use a GPU

```python
f = Flow().add(
    uses='jinahub+docker://DPRReaderRanker/gpu',
    uses_with={'device': 'cuda'},
    gpus='all',
    volumes='/your/home/dir/.cache/huggingface:/root/.cache/huggingface' 
)
```

## See Also
- [DPRTextEncoder](https://hub.jina.ai/executor/awl0jxog)

## Reference

- [DPR paper](https://arxiv.org/abs/2004.04906)
- [Huggingface transformers DPR model documentation](https://huggingface.co/transformers/model_doc/dpr.html)

<!-- version=v0.2 -->
