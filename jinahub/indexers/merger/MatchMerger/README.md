# MatchMerger

**MatchMerger** Merges the results of shards by appending all matches. Assume you have 20 shards and use `top-k=10`, you will get 200 results in the merger.
The `MatchMerger` is used in the `uses_after` attribute when adding an `Executor` to the `Flow`.
