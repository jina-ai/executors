__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import gzip
from typing import Optional, Dict, List

import numpy as np
from jina import Executor, DocumentArray, requests, Document

from jina_commons import get_logger
from jina_commons.indexers.dump import import_vectors


class FaissSearcher(Executor):
    """Faiss-powered vector indexer

    For more information about the Faiss supported parameters and installation problems, please consult:
        - https://github.com/facebookresearch/faiss

    .. note::
        Faiss package dependency is only required at the query time.

    :param index_key: index type supported by ``faiss.index_factory``
    :param train_filepath: the training data file path, e.g ``faiss.tgz`` or `faiss.npy`. The data file is expected
        to be either `.npy` file from `numpy.save()` or a `.tgz` file from `NumpyIndexer`. If none is provided, `indexed` data will be used
        to train the Indexer (In that case, one must be careful when sharding is enabled, because every shard will be trained with its own part of data).
        The data will only be loaded if `requires_training` is set to True.
    :param max_num_training_points: Optional argument to consider only a subset of training points to training data from `train_filepath`.
        The points will be selected randomly from the available points
    :param requires_training: Boolean flag indicating if the index type requires training to be run before building index.
    :param metric: 'l2' or 'inner_product' accepted. Determines which distances to optimize by FAISS. l2...smaller is better, inner_product...larger is better
    :param normalize: whether or not to normalize the vectors e.g. for the cosine similarity https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
    :param nprobe: Number of clusters to consider at search time.
    :param is_distance: Boolean flag that describes if distance metric need to be reinterpreted as similarities.

    .. highlight:: python
    .. code-block:: python
        # generate a training file in `.tgz`
        import gzip
        import numpy as np
        from jina.executors.indexers.vector.faiss import FaissIndexer

        train_filepath = 'faiss_train.tgz'
        train_data = np.random.rand(10000, 128)
        with gzip.open(train_filepath, 'wb', compresslevel=1) as f:
            f.write(train_data.astype('float32'))
        indexer = FaissIndexer('PCA64,FLAT', train_filepath)

        # generate a training file in `.npy`
        train_filepath = 'faiss_train'
        np.save(train_filepath, train_data)
        indexer = FaissIndexer('PCA64,FLAT', train_filepath)
    """

    def __init__(
        self,
        index_key: str,
        train_filepath: Optional[str] = None,
        max_num_training_points: Optional[int] = None,
        requires_training: bool = True,
        metric: str = 'l2',
        normalize: bool = False,
        nprobe: int = 1,
        dump_path: Optional[str] = None,
        default_traversal_paths: List[str] = ['r'],
        is_distance: bool = False,
        default_top_k: int = 5,
        on_gpu: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index_key = index_key
        self.requires_training = requires_training
        self.train_filepath = train_filepath if self.requires_training else None
        self.max_num_training_points = max_num_training_points
        self.metric = metric
        self.normalize = normalize
        self.nprobe = nprobe
        self.on_gpu = on_gpu

        self.default_top_k = default_top_k
        self.default_traversal_paths = default_traversal_paths
        self.is_distance = is_distance

        self.logger = get_logger(self)

        dump_path = dump_path or kwargs.get('runtime_args').get('dump_path')
        if dump_path is not None:
            self.logger.info('Start building "AnnoyIndexer" from dump data')
            ids, vecs = import_vectors(dump_path, str(self.runtime_args.pea_id))
            self._ids = np.array(list(ids))
            self._ext2int = {v: i for i, v in enumerate(self._ids)}
            self._vecs = np.array(list(vecs))
            self.num_dim = self._vecs.shape[1]
            self.dtype = self._vecs.dtype
            self.index = self._build_index(self._vecs)
        else:
            self.logger.warning(
                'No data loaded in "AnnoyIndexer". Use .rolling_update() to re-initialize it...'
            )

    def device(self):
        """
        Set the device on which the executors using :mod:`faiss` library will be running.

        ..notes:
            In the case of using GPUs, we only use the first gpu from the visible gpus. To specify which gpu to use,
            please use the environment variable `CUDA_VISIBLE_DEVICES`.
        """
        import faiss

        # For now, consider only one GPU, do not distribute the index
        return faiss.StandardGpuResources() if self.on_gpu else None

    def to_device(self, index, *args, **kwargs):
        """Load the model to device."""
        import faiss

        device = self.device()
        return (
            faiss.index_cpu_to_gpu(device, 0, index, None)
            if device is not None
            else index
        )

    def _build_index(self, vecs: 'np.ndarray'):
        """Build an advanced index structure from a numpy array.

        :param vecs: numpy array containing the vectors to index
        """
        import faiss

        metric = faiss.METRIC_L2
        if self.metric == 'inner_product':
            self.logger.warning(
                'inner_product will be output as distance instead of similarity.'
            )
            metric = faiss.METRIC_INNER_PRODUCT
        if self.metric not in {'inner_product', 'l2'}:
            self.logger.warning(
                'Invalid distance metric for Faiss index construction. Defaulting to l2 distance'
            )

        index = self.to_device(
            index=faiss.index_factory(self.num_dim, self.index_key, metric)
        )

        if self.requires_training:
            if self.train_filepath:
                train_data = self._load_training_data(self.train_filepath)
            else:
                self.logger.info(f'Taking indexed data as training points')
                train_data = vecs
            if train_data is None:
                self.logger.warning(
                    'Loading training data failed. some faiss indexes require previous training.'
                )
            else:
                if self.max_num_training_points:
                    self.logger.warning(
                        f'From train_data with num_points {train_data.shape[0]}, '
                        f'sample {self.max_num_training_points} points'
                    )
                    random_indices = np.random.choice(
                        train_data.shape[0],
                        size=min(self.max_num_training_points, train_data.shape[0]),
                        replace=False,
                    )
                    train_data = train_data[random_indices, :]
                train_data = train_data.astype(np.float32)
                if self.normalize:
                    faiss.normalize_L2(train_data)
                self._train(index, train_data)

        self._build_partial_index(vecs, index)
        index.nprobe = self.nprobe
        return index

    def _build_partial_index(self, vecs: 'np.ndarray', index):
        vecs = vecs.astype(np.float32)
        if self.normalize:
            from faiss import normalize_L2

            normalize_L2(vecs)
        index.add(vecs)

        return

    @requests(on='/search')
    def search(
        self, docs: DocumentArray, parameters: Optional[Dict] = None, *args, **kwargs
    ):
        """Find the top-k vectors with smallest ``metric`` and return their ids in ascending order.

        :param docs: the DocumentArray containing the documents to search with
        :param parameters: the parameters for the request
        """
        if not hasattr(self, 'index'):
            self.logger.warning('Querying against an empty Index')
            return

        if parameters is None:
            parameters = {}

        top_k = parameters.get('top_k', self.default_top_k)
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        query_docs = docs.traverse_flat(traversal_paths)

        vecs = np.array(query_docs.get_attributes('embedding'))

        if self.normalize:
            from faiss import normalize_L2

            normalize_L2(vecs)
        dists, ids = self.index.search(vecs, top_k)
        if self.metric == 'inner_product':
            dists = 1 - dists
        for doc_idx, matches in enumerate(zip(ids, dists)):
            for m_info in zip(*matches):
                idx, dist = m_info
                match = Document(id=self._ids[idx], embedding=self._vecs[idx])
                if self.is_distance:
                    match.scores[self.metric] = dist
                else:
                    if self.metric == 'inner_product':
                        match.scores[self.metric] = 1 - dist
                    else:
                        match.scores[self.metric] = 1 / (1 + dist)

                query_docs[doc_idx].matches.append(match)

    def _train(self, index, data: 'np.ndarray', *args, **kwargs) -> None:
        _num_samples, _num_dim = data.shape
        if not self.num_dim:
            self.num_dim = _num_dim
        if self.num_dim != _num_dim:
            raise ValueError(
                'training data should have the same number of features as the index, {} != {}'.format(
                    self.num_dim, _num_dim
                )
            )
        self.logger.info(
            f'Training faiss Indexer with {_num_samples} points of {self.num_dim}'
        )

        index.train(data)

    def _load_training_data(self, train_filepath: str) -> 'np.ndarray':
        self.logger.info(f'Loading training data from {train_filepath}')
        result = None
        try:
            result = self._load_gzip(train_filepath)
        except Exception as e:
            self.logger.error(
                'Loading training data from gzip failed, filepath={}, {}'.format(
                    train_filepath, e
                )
            )

        if result is None:
            try:
                result = np.load(train_filepath)
                if isinstance(result, np.lib.npyio.NpzFile):
                    self.logger.warning(
                        '.npz format is not supported. Please save the array in .npy format.'
                    )
                    result = None
            except Exception as e:
                self.logger.error(
                    'Loading training data with np.load failed, filepath={}, {}'.format(
                        train_filepath, e
                    )
                )

        if result is None:
            try:
                # Read from binary file:
                with open(train_filepath, 'rb') as f:
                    result = f.read()
            except Exception as e:
                self.logger.error(
                    'Loading training data from binary file failed, filepath={}, {}'.format(
                        train_filepath, e
                    )
                )
        return result

    def _load_gzip(self, abspath: str, mode='rb') -> Optional['np.ndarray']:
        try:
            self.logger.info(f'loading index from {abspath}...')
            with gzip.open(abspath, mode) as fp:
                return np.frombuffer(fp.read(), dtype=self.dtype).reshape(
                    [-1, self.num_dim]
                )
        except EOFError:
            self.logger.error(
                f'{abspath} is broken/incomplete, perhaps forgot to ".close()" in the last usage?'
            )

    @property
    def size(self):
        """Return the nr of elements in the index"""
        if hasattr(self, '_ids'):
            return len(self._ids)
        else:
            return 0

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """Retrieve the embeddings of the documents (if they are in the index)

        :param docs: the DocumentArray containing the documents to search with
        """
        for doc in docs:
            try:
                doc.embedding = self._vecs[self._ext2int[doc.id]]
            except Exception as e:
                self.logger.warning(
                    f'Document with id {doc.id} could not be processed. Error: {e}'
                )
