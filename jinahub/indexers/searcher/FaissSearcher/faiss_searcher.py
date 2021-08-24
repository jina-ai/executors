__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Iterable, Optional, Dict, List
import numpy as np
import faiss

from jina import Executor, DocumentArray, requests, Document
from jina.helper import batch_iterator
from jina_commons import get_logger
from jina_commons.indexers.dump import import_vectors


class FaissSearcher(Executor):
    """Faiss-powered vector indexer

    For more information about the Faiss supported parameters and installation problems, please consult:
        - https://github.com/facebookresearch/faiss

    .. note::
        Faiss package dependency is only required at the query time.

    :param index_key: index type supported by ``faiss.index_factory``
    :param trained_index_file: the index file dumped from a trained index, e.g., ``faiss.index``. If none is provided, `indexed` data will be used
        to train the Indexer (In that case, one must be careful when sharding is enabled, because every shard will be trained with its own part of data).
    :param max_num_training_points: Optional argument to consider only a subset of training points to training data from `train_filepath`.
        The points will be selected randomly from the available points
    :param prefetch_size: the number of data to pre-load into RAM
    :param requires_training: Boolean flag indicating if the index type requires training to be run before building index.
    :param metric: 'l2' or 'inner_product' accepted. Determines which distances to optimize by FAISS. l2...smaller is better, inner_product...larger is better
    :param normalize: whether or not to normalize the vectors e.g. for the cosine similarity https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
    :param nprobe: Number of clusters to consider at search time.
    :param is_distance: Boolean flag that describes if distance metric need to be reinterpreted as similarities.
    :param make_direct_map: Boolean flag that describes if direct map has to be computed after building the index. Useful if you need to call `fill_embedding` endpoint and reconstruct vectors
        by id

    .. highlight:: python
    .. code-block:: python
        # generate a training file in `.tgz`
        import gzip
        import numpy as np
        from jina.executors.indexers.vector.faiss import FaissIndexer

        import faiss
        trained_index_file = os.path.join(os.environ['TEST_WORKSPACE'], 'faiss.index')
        train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
        faiss_index = faiss.index_factory(10, 'IVF10,PQ2')
        faiss_index.train(train_data)
        faiss.write_index(faiss_index, trained_index_file)

        searcher = FaissSearcher('PCA64,FLAT', trained_index_file=trained_index_file)

    """

    def __init__(
        self,
        index_key: str = 'Flat',
        trained_index_file: Optional[str] = None,
        max_num_training_points: Optional[int] = None,
        requires_training: bool = True,
        metric: str = 'l2',
        normalize: bool = False,
        nprobe: int = 1,
        dump_path: Optional[str] = None,
        prefetch_size: Optional[int] = 1000,
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
        self.trained_index_file = trained_index_file

        self.max_num_training_points = max_num_training_points
        self.prefetch_size = prefetch_size
        self.metric = metric
        self.normalize = normalize
        self.nprobe = nprobe
        self.on_gpu = on_gpu

        self.default_top_k = default_top_k
        self.default_traversal_paths = default_traversal_paths
        self.is_distance = is_distance
        self._doc_id_to_offset = {}

        self.logger = get_logger(self)

        dump_path = dump_path or kwargs.get('runtime_args').get('dump_path')
        if dump_path is not None:
            self.logger.info('Start building "FaissIndexer" from dump data')
            ids_iter, vecs_iter = import_vectors(
                dump_path, str(self.runtime_args.pea_id)
            )
            self._ids = np.array(list(ids_iter))
            self._doc_id_to_offset = {v: i for i, v in enumerate(self._ids)}

            self._prefetch_data = []
            if self.prefetch_size and self.prefetch_size > 0:
                for _ in range(prefetch_size):
                    try:
                        self._prefetch_data.append(next(vecs_iter))
                    except StopIteration:
                        break
            else:
                self._prefetch_data = list(vecs_iter)

            self.num_dim = self._prefetch_data[0].shape[0]
            self.dtype = self._prefetch_data[0].dtype
            self.index = self._build_index(vecs_iter)
        else:
            self.logger.warning(
                'No data loaded in "FaissIndexer". Use .rolling_update() to re-initialize it...'
            )

    def device(self):
        """
        Set the device on which the executors using :mod:`faiss` library will be running.

        ..notes:
            In the case of using GPUs, we only use the first gpu from the visible gpus. To specify which gpu to use,
            please use the environment variable `CUDA_VISIBLE_DEVICES`.
        """

        # For now, consider only one GPU, do not distribute the index
        return faiss.StandardGpuResources() if self.on_gpu else None

    def to_device(self, index, *args, **kwargs):
        """Load the model to device."""

        if self.on_gpu and ('PQ64' in self.index_key):
            co = faiss.GpuClonerOptions()

            # Due to the limited temporary memory, we must set the lookup tables to
            # 16 bit float while using 64-byte PQ
            co.useFloat16 = True
        else:
            co = None

        device = self.device()
        return (
            faiss.index_cpu_to_gpu(device, 0, index, co)
            if device is not None
            else index
        )

    def _init_index(self):

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

        if self.trained_index_file and os.path.exists(self.trained_index_file):
            index = faiss.read_index(self.trained_index_file)
            assert index.metric_type == metric
            assert index.ntotal == 0
            assert index.d == self.num_dim
            assert index.is_trained
        else:
            index = faiss.index_factory(self.num_dim, self.index_key, metric)

        index.nprobe = self.nprobe

        index = self.to_device(index)

        return index

    def _build_index(self, vecs_iter: Iterable['np.ndarray']):
        """Build an advanced index structure from a numpy array.

        :param vecs_iter: iterator of numpy array containing the vectors to index
        """

        index = self._init_index()

        if self.requires_training and (not index.is_trained):
            self.logger.info(f'Taking indexed data as training points')
            if self.max_num_training_points is None:
                self._prefetch_data.extend(list(vecs_iter))
            else:
                while len(self._prefetch_data) < self.max_num_training_points:
                    try:
                        self._prefetch_data.append(next(vecs_iter))
                    except:
                        break

            train_data = np.stack(self._prefetch_data)

            if (
                self.max_num_training_points
                and self.max_num_training_points < train_data.shape[0]
            ):
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

            self.logger.info(f'Training Faiss {self.index_key} indexer...')
            train_data = train_data.astype(np.float32)
            if self.normalize:
                faiss.normalize_L2(train_data)
            self._train(index, train_data)

            self.logger.info(
                f'Dumping the trained Faiss index to {self.trained_index_file}'
            )
            if self.on_gpu:
                index = faiss.index_gpu_to_cpu(index)
            if self.trained_index_file:
                if os.path.exists(self.trained_index_file):
                    self.logger.warning(
                        f'We are going to overwrite the index file located at {self.trained_index_file}'
                    )
                faiss.write_index(index, self.trained_index_file)

        # TODO: Experimental features
        # if 'IVF' in self.index_key:
        #     # Support for searching several inverted lists in parallel (parallel_mode != 0)
        #     self.logger.info(
        #         'We will setting `parallel_mode=1` to supporting searching several inverted lists in parallel'
        #     )
        #     index.parallel_mode = 1

        self.logger.info(f'Building the faiss {self.index_key} index...')
        self._build_partial_index(vecs_iter, index)

        return index

    def _build_partial_index(self, vecs_iter: Iterable['np.ndarray'], index):
        if len(self._prefetch_data) > 0:
            vecs = np.stack(self._prefetch_data).astype(np.float32)
            self._index(vecs, index)
            self._prefetch_data.clear()

        for batch_data in batch_iterator(vecs_iter, self.prefetch_size):
            batch_data = list(batch_data)
            if len(batch_data) == 0:
                break
            vecs = np.stack(batch_data).astype(np.float32)
            self._index(vecs, index)

        return

    def _index(self, vecs: 'np.ndarray', index):
        if self.normalize:
            from faiss import normalize_L2

            normalize_L2(vecs)
        index.add(vecs)

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

        top_k = int(parameters.get('top_k', self.default_top_k))
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
                match = Document(id=self._ids[idx])
                if self.is_distance:
                    match.scores[self.metric] = dist
                else:
                    if self.metric == 'inner_product':
                        match.scores[self.metric] = 1 - dist
                    else:
                        match.scores[self.metric] = 1 / (1 + dist)

                query_docs[doc_idx].matches.append(match)

    @requests(on='/train')
    def train(self, parameters: Dict, **kwargs):
        """Train the index

        :param parameters: a dictionary containing the parameters for the training
        """

        train_filepath = parameters.get('train_filepath')
        if train_filepath is None:
            raise ValueError(f'No "train_filepath" provided for {self}')

        max_num_training_points = parameters.get(
            'max_num_training_points', self.max_num_training_points
        )
        trained_index_file = parameters.get(
            'trained_index_file', self.trained_index_file
        )
        if not trained_index_file:
            raise ValueError(
                'the trained index file path is not provided to dump trained index'
            )

        train_data = self._load_training_data(train_filepath)

        self.num_dim = train_data.shape[1]
        self.dtype = train_data.dtype

        index = self._init_index()

        if train_data is None:
            self.logger.warning(
                'Loading training data failed. some faiss indexes require previous training.'
            )
        else:
            train_data = train_data.astype(np.float32)
            if (
                max_num_training_points
                and max_num_training_points < train_data.shape[0]
            ):
                self.logger.warning(
                    f'From train_data with num_points {train_data.shape[0]}, '
                    f'sample {max_num_training_points} points'
                )
                random_indices = np.random.choice(
                    train_data.shape[0],
                    size=min(self.max_num_training_points, train_data.shape[0]),
                    replace=False,
                )
                train_data = train_data[random_indices, :]

            if self.normalize:
                faiss.normalize_L2(train_data)
            self._train(index, train_data)

            self.logger.info(f'Dumping the trained Faiss index to {trained_index_file}')

            if trained_index_file:
                if os.path.exists(trained_index_file):
                    self.logger.warning(
                        f'We are going to overwrite the index file located at {trained_index_file}'
                    )
                faiss.write_index(index, trained_index_file)

            

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

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: Optional[DocumentArray], **kwargs):
        if docs is None:
            return
        for doc in docs:
            if doc.id in self._doc_id_to_offset:
                try:
                    reconstruct_embedding = self.index.reconstruct(self._doc_id_to_offset[doc.id])
                    doc.embedding = np.array(
                        reconstruct_embedding
                    )
                except RuntimeError as exception:
                    self.logger.warning(f'Trying to reconstruct from document id failed. Most likely the index built '
                                        f'from index key {self.index_key} does not support this operation. {repr(exception)}')
            else:
                self.logger.debug(f'Document {doc.id} not found in index')

    @property
    def size(self):
        """Return the nr of elements in the index"""
        return len(self._doc_id_to_offset)
