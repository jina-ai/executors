from jina import Executor, DocumentArray, requests


class LightGBMRanker(Executor):
    """LightGBMRanker learning to rank"""
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
