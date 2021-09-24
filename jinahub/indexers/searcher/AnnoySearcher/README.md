# AnnoySearcher

**AnnoySearcher** is an Annoy-powered vector-based similarity searcher. Annoy stands for "Approximate Nearest Neighbors Oh Yeah", and is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mmapped into memory so that many processes may share the same data.

For more information, refer to the GitHub repo for [Spotify's Annoy](https://github.com/spotify/annoy).



## Usage

Check [tests](tests) for an example on how to use it.

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data. Rather they are write-once classes, which take as data source a `dump_path`. 

This can be provided in different ways:

- in the YAML definition
  
```yaml
jtype: AnnoySearcher
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [docs](https://docs.jina.ai/advanced/experimental/indexers/).

The folder needs to contain the data exported from your Indexer. Again, see [docs](https://docs.jina.ai/advanced/experimental/indexers/).

## Reference

- https://github.com/spotify/annoy

<!-- version=v0.1 -->
