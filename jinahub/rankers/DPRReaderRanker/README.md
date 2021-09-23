# DPRReaderRanker

 **DPRReaderRanker** performs a re-ranking of the matches for each document (question), as well as the answer spans extraction for each match. It uses the DPR Reader model to re-rank documents based on cross-attention between the question and the potential answer passages.

The **DPR** model was originally proposed in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906).


## Reference


- [DPR paper](https://arxiv.org/abs/2004.04906)
- [Huggingface transformers DPR model documentation](https://huggingface.co/transformers/model_doc/dpr.html)
