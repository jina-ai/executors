# SimpleRanker

SimpleRanker aggregates the score of matched chunks (matches that are chunks) to the
score of the parent document of the matched chunks. The Document's matches are then
replaced by matches based on the parent documents of the chunks - they contain an
`id` and the aggregated score only.

This ranker is used to "bubble-up" the scores of matched chunks to the scores
of the parent document. As an example, consider an application where we are matching
song lyrics to an input query (like in the [multires lyrics search example](https://github.com/jina-ai/examples/tree/master/multires-lyrics-search)). During indexing we break down all the song lyrics into sentences.

During querying, we first match lyric sentences to the query, and
then use this ranker to produce matching songs (whole songs/lyrics, not just
sentences) for the query. Since the matches that this ranker produces contain only
document id, we add a final step where an indexer is used to retrieve the song
contents base on the id.
## Reference
- See the [multires lyrics search example](https://github.com/jina-ai/examples/tree/master/multires-lyrics-search) for example usage
