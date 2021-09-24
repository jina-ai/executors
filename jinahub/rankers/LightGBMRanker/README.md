# LightGBMRanker

**LightGBMRanker** uses the [LightGBM](https://github.com/microsoft/LightGBM) library to implement learning-to-rank.

`LightGBMRanker` retrieves `query_features`, `match_features` and `relevance_label` stored inside `Document` object from `DocumentArray`, and builds a feature-label dataset to train the model.


## Reference
Refer to LightGBM [documentation](https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank) to learn how to use LightGBM to train a ranker.

<!-- version=v0.1 -->
