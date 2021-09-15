# CatboostRanker

**CatboostRanker** uses the [CatBoost](https://catboost.ai/) library to perform learning-to-rank.

`CatboostRanker` retrieves `query_features`, `match_features` and `relevance_label` stored inside `Document` object from `DocumentArray`, and builds a feature-label dataset to train the model.


## Reference

Refer to CatBoost [tutorial](https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb) to learn how to use CatBoost to train a ranker.
