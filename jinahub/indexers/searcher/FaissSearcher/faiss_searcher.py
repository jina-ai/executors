__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import gzip
import os
import pickle
from datetime import datetime
from typing import Callable, Dict, Generator, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from jina import Document, DocumentArray, Executor, requests
from jina.helper import batch_iterator
from jina_commons import get_logger
from jina_commons.indexers.dump import import_vectors

GENERATOR_DELTA = Generator[
    Tuple[str, Optional[np.ndarray], Optional[datetime]], None, None
]

DELETE_MARKS_FILENAME = 'delete_marks.bin'
DOC_IDS_FILENAME = 'doc_ids.bin'
FAISS_INDEX_FILENAME = 'faiss.bin'


class FaissSearcher(Executor):
    """Faiss-powered vector indexer

    For more information about the Faiss
    supported parameters and installation problems, please consult:
        - https://github.com/facebookresearch/faiss

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
        # exhaustive search, which corresponds to what NumpySearcher was doing
        index_key: str = 'Flat',
        trained_index_file: Optional[str] = None,
        max_num_training_points: Optional[int] = None,
        metric: str = 'cosine',
        nprobe: int = 1,
        dump_path: Optional[str] = None,
        dump_func: Optional[Callable] = None,
        prefetch_size: Optional[int] = 512,
        default_traversal_paths: List[str] = ['r'],
        is_distance: bool = True,
        default_top_k: int = 5,
        on_gpu: bool = False,
        *args,
        **kwargs,
    ):
        """
        :param trained_index_file: the index file dumped from a trained
            index, e.g., ``faiss.index``. If none is provided, `indexed` data will be used
            to train the Indexer (In that case, one must be careful when sharding
            is enabled, because every shard will be trained with its own part of data).
        :param index_key: index type supported
            by ``faiss.index_factory``
        :param train_filepath: the training data file path,
            e.g ``faiss.tgz`` or `faiss.npy`. The data file is expected
            to be either `.npy` file from `numpy.save()` or a `.tgz` file
            from `NumpyIndexer`. If none is provided, `indexed` data will be used
            to train the Indexer (In that case, one must be careful when sharding
            is enabled, because every shard will be trained with its own part of data).
            The data will only be loaded if the index type requires training to be run
            before building index.
        :param max_num_training_points: Optional argument to consider only a subset of
        training points to training data from `train_filepath`.
            The points will be selected randomly from the available points
        :param prefetch_size: the number of data to pre-load into RAM
        :param metric: 'euclidean', 'cosine' or 'inner_product' accepted. Determines which distances to
            optimize by FAISS. euclidean...smaller is better, cosine...larger is better
        :param nprobe: Number of clusters to consider at search time.
        :param is_distance: Boolean flag that describes if distance metric need to be
            reinterpreted as similarities.
        """
        super().__init__(*args, **kwargs)
        self.last_timestamp = datetime.min
        self.index_key = index_key
        self.trained_index_file = trained_index_file

        self.max_num_training_points = max_num_training_points
        self.prefetch_size = prefetch_size
        self.metric = metric

        self.normalize = False
        if self.metric == 'cosine':
            self.normalize = True

        self.nprobe = nprobe
        self.on_gpu = on_gpu

        self.default_top_k = default_top_k
        self.default_traversal_paths = default_traversal_paths
        self.is_distance = is_distance

        self._doc_ids = []
        self._doc_id_to_offset = {}

        self._is_deleted = []
        self._prefetch_data = []
        self._faiss_index = None
        self.logger = get_logger(self)

        dump_path = dump_path or kwargs.get('runtime_args', {}).get('dump_path')
        if dump_path or dump_func:
            self._load_dump(dump_path, dump_func, prefetch_size, **kwargs)
        else:
            self._load(self.workspace)

    def _load_dump(self, dump_path, dump_func, prefetch_size, **kwargs):
        if dump_path is not None:
            self.logger.info(
                f'Start building "FaissIndexer" from dump data {dump_path}'
            )
            ids_iter, vecs_iter = import_vectors(
                dump_path, str(self.runtime_args.pea_id)
            )
            iterator = zip(ids_iter, vecs_iter)
        elif dump_func is not None:
            iterator = dump_func(shard_id=self.runtime_args.pea_id)
        else:
            self.logger.warning(
                'No "dump_path" or "dump_func" passed to "FaissIndexer".'
                ' Use .rolling_update() to re-initialize it...'
            )
            return

        if iterator is not None:
            iterator = self._iterate_vectors_and_save_ids(iterator)
            self._prefetch_data = []
            if self.prefetch_size and self.prefetch_size > 0:
                for _ in range(prefetch_size):
                    try:
                        self._prefetch_data.append(next(iterator))
                    except StopIteration:
                        break
            else:
                self._prefetch_data = list(iterator)

            if len(self._prefetch_data) == 0:
                return

            self.num_dim = self._prefetch_data[0].shape[0]
            self.dtype = self._prefetch_data[0].dtype
            self._build_index(iterator)

    def _iterate_vectors_and_save_ids(self, iterator):
        for position, id_vector in enumerate(iterator):
            id_ = id_vector[0]
            vector = id_vector[1]
            self._doc_ids.append(id_)
            self._doc_id_to_offset[id_] = position
            self._is_deleted.append(0)
            if vector is not None:
                # this should already be a np.array, NOT bytes
                yield vector
            else:
                yield None

    def device(self):
        """
        Set the device on which the executors using :mod:`faiss` library
         will be running.

        ..notes:
            In the case of using GPUs, we only use the first gpu from the
            visible gpus. To specify which gpu to use,
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

    def _init_faiss_index(self, num_dim: int, trained_index_file: Optional[str] = None):
        """Initialize a Faiss indexer instance"""
        if trained_index_file and os.path.exists(trained_index_file):
            index = faiss.read_index(trained_index_file)
            assert index.metric_type == self.metric_type
            assert index.ntotal == 0
            assert not hasattr(self, 'num_dim') or index.d == self.num_dim
            assert index.is_trained
        else:
            index = faiss.index_factory(num_dim, self.index_key, self.metric_type)

        self._faiss_index = self.to_device(index)
        self._faiss_index.nprobe = self.nprobe

    def _build_index(self, vecs_iter: Iterable['np.ndarray']):
        """Build an advanced index structure from a numpy array.

        :param vecs_iter: iterator of numpy array containing the vectors to index
        """

        self._init_faiss_index(self.num_dim, trained_index_file=self.trained_index_file)

        if not self._faiss_index.is_trained:
            self.logger.info('Taking indexed data as training points...')
            if self.max_num_training_points is None:
                self._prefetch_data.extend(list(vecs_iter))
            else:
                self.logger.info('Taking indexed data as training points')
                while (
                    self.max_num_training_points
                    and len(self._prefetch_data) < self.max_num_training_points
                ):
                    try:
                        self._prefetch_data.append(next(vecs_iter))
                    except Exception as _:  # noqa: F841
                        break

            if len(self._prefetch_data) == 0:
                return

            train_data = np.stack(self._prefetch_data)
            train_data = train_data.astype(np.float32)

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

            self.logger.info('Training Faiss indexer...')

            if self.normalize:
                faiss.normalize_L2(train_data)
            self._train(train_data)

        self.logger.info('Building the Faiss index...')
        self._build_partial_index(vecs_iter)

    def _build_partial_index(self, vecs_iter: Iterable['np.ndarray']):
        if len(self._prefetch_data) > 0:
            vecs = np.stack(self._prefetch_data).astype(np.float32)
            self._index(vecs)
            self._prefetch_data.clear()

        for batch_data in batch_iterator(vecs_iter, self.prefetch_size):
            batch_data = list(batch_data)
            if len(batch_data) == 0:
                break
            vecs = np.stack(batch_data).astype(np.float32)
            self._index(vecs)

    def _index(self, vecs: 'np.ndarray'):
        if self.normalize:
            from faiss import normalize_L2

            normalize_L2(vecs)
        self._faiss_index.add(vecs)

    @requests(on='/search')
    def search(
        self,
        docs: Optional[DocumentArray],
        parameters: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        """Find the top-k vectors with smallest
        ``metric`` and return their ids in ascending order.

        :param docs: the DocumentArray containing the documents to search with
        :param parameters: the parameters for the request

        :return: Attaches matches to the Documents sent as inputs, with the id of the
            match, and its embedding.
        """
        if docs is None:
            return
        if not hasattr(self, '_faiss_index'):
            self.logger.warning('Querying against an empty Index')
            return

        if parameters is None:
            parameters = {}

        top_k = int(parameters.get('top_k', self.default_top_k))
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        # expand topk number guarantee to return topk results
        # TODO WARNING: maybe this would degrade the query speed
        expand_topk = top_k + self.deleted_count

        query_docs = docs.traverse_flat(traversal_paths)

        vecs = np.array(query_docs.get_attributes('embedding')).astype(np.float32)

        if self.normalize:
            from faiss import normalize_L2

            normalize_L2(vecs)

        dists, ids = self._faiss_index.search(vecs, expand_topk)

        if self.metric in ['cosine', 'inner_product']:
            dists = 1 - dists

        for doc_idx, matches in enumerate(zip(ids, dists)):
            count = 0
            for m_info in zip(*matches):
                idx, dist = m_info

                # this is related with the issue of faiss
                if idx < 0:
                    continue

                doc_id = self._doc_ids[idx]

                if self.is_deleted(idx):
                    continue

                match = Document(id=doc_id)
                if self.is_distance:
                    match.scores[self.metric] = dist
                else:
                    if self.metric in ['cosine', 'inner_product']:
                        match.scores[self.metric] = 1 - dist
                    else:
                        match.scores[self.metric] = 1 / (1 + dist)

                query_docs[doc_idx].matches.append(match)

                # early stop as topk results are ready
                count += 1
                if count >= top_k:
                    break

    @requests(on='/index')
    def index(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Index the Documents' embeddings.
        :param docs: `Document` with same shaped `.embedding`.
        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. The only supported key is
            `traversal_paths`.
        """

        if docs is None:
            return

        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        try:
            ids = flat_docs.get_attributes('id')
            vecs = flat_docs.embeddings
            self._append_vecs_and_ids(ids, vecs)
        except Exception as ex:
            self.logger.error(f'failed to index docs, {ex}')
            raise ex

    @requests(on='/save')
    def save(self, parameters: Dict, **kwargs):
        """
        Save a snapshot of the current indexer
        """

        target_path = (
            parameters['target_path'] if 'target_path' in parameters else self.workspace
        )

        os.makedirs(target_path, exist_ok=True)

        # dump faiss index
        faiss.write_index(
            self._faiss_index, os.path.join(target_path, FAISS_INDEX_FILENAME)
        )

        with open(os.path.join(target_path, DOC_IDS_FILENAME), "wb") as fp:
            pickle.dump(self._doc_ids, fp)

        with open(os.path.join(target_path, DELETE_MARKS_FILENAME), "wb") as fp:
            pickle.dump(self._is_deleted, fp)

    def _faiss_index_exist(self, folder_path: str):
        index_path = os.path.join(folder_path, FAISS_INDEX_FILENAME)
        return os.path.exists(index_path)

    def _load(self, from_path: Optional[str] = None):
        from_path = from_path if from_path else self.workspace
        self.logger.info(f'Try to restore indexer from {from_path}...')
        try:
            with open(os.path.join(from_path, DOC_IDS_FILENAME), 'rb') as fp:
                self._doc_ids = pickle.load(fp)
                self._doc_id_to_offset = {v: i for i, v in enumerate(self._doc_ids)}

            with open(os.path.join(from_path, DELETE_MARKS_FILENAME), 'rb') as fp:
                self._is_deleted = pickle.load(fp)

            index = faiss.read_index(os.path.join(from_path, FAISS_INDEX_FILENAME))
            assert index.metric_type == self.metric_type
            assert index.is_trained
            self.num_dim = index.d
            self._faiss_index = self.to_device(index)
            self._faiss_index.nprobe = self.nprobe
        except FileNotFoundError:
            self.logger.warning(
                'None snapshot is found, you should build the indexer from scratch'
            )
            return False
        except Exception as ex:
            raise ex

        return True

    def train(
        self,
        docs: Optional[DocumentArray] = None,
        parameters: Optional[Dict] = None,
        **kwargs,
    ):
        """Train the index

        :param docs: docs for training index
        :param parameters: a dictionary containing the parameters for the training
        """
        if docs is None:
            return

        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        try:
            train_data = flat_docs.embeddings
        except Exception as ex:
            self.logger.error(f'failed to train the index, {ex}')
            raise ex

        self.num_dim = train_data.shape[1]
        self.dtype = train_data.dtype

        train_data = train_data.astype(np.float32)

        self._init_faiss_index(self.num_dim)

        if self.normalize:
            faiss.normalize_L2(train_data)
        self._train(train_data)

        index_data = parameters.get('index_data', True)
        if index_data:
            self.index(docs=docs, parameters=parameters)

    def _train(self, data: 'np.ndarray', *args, **kwargs) -> None:
        _num_samples, _num_dim = data.shape
        if not self.num_dim:
            self.num_dim = _num_dim
        if self.num_dim != _num_dim:
            raise ValueError(
                'training data should have the same '
                'number of features as the index, {} != {}'.format(
                    self.num_dim, _num_dim
                )
            )
        self.logger.info(
            f'Training faiss Indexer with {_num_samples} points of {self.num_dim}'
        )

        self._faiss_index.train(data)

    def _load_training_data(self, train_filepath: str) -> 'np.ndarray':
        self.logger.info(f'Loading training data from {train_filepath}')
        result = None

        try:
            result = np.load(train_filepath)
            if isinstance(result, np.lib.npyio.NpzFile):
                self.logger.warning(
                    '.npz format is not supported. Please save the array in .npy '
                    'format.'
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
                result = np.load(train_filepath)
                if isinstance(result, np.lib.npyio.NpzFile):
                    self.logger.warning(
                        '.npz format is not supported. '
                        'Please save the array in .npy format.'
                    )
                    result = None
            except Exception as e:
                self.logger.error(
                    'Loading training data with np.load failed, filepath={}, '
                    '{}'.format(train_filepath, e)
                )

        if result is None:
            try:
                # Read from binary file:
                with open(train_filepath, 'rb') as f:
                    result = f.read()
            except Exception as e:
                self.logger.error(
                    'Loading training data from binary'
                    ' file failed, filepath={}, {}'.format(train_filepath, e)
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
                f'{abspath} is broken/incomplete, '
                f'perhaps forgot to ".close()" in the last usage?'
            )

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: Optional[DocumentArray], **kwargs):
        if docs is None:
            return
        for doc in docs:
            if doc.id in self._doc_id_to_offset:
                try:
                    reconstruct_embedding = self._faiss_index.reconstruct(
                        self._doc_id_to_offset[doc.id]
                    )
                    doc.embedding = np.array(reconstruct_embedding)
                except RuntimeError as exception:
                    self.logger.warning(
                        f'Trying to reconstruct from '
                        f'document id failed. Most '
                        f'likely the index built '
                        f'from index key {self.index_key} \
                         does not support this '
                        f'operation. {repr(exception)}'
                    )
            else:
                self.logger.debug(f'Document {doc.id} not found in index')

    @property
    def size(self):
        """Return the nr of elements in the index"""
        return len(self._doc_ids) - self.deleted_count

    @property
    def deleted_count(self):
        return sum(self._is_deleted)

    @property
    def metric_type(self):
        metric_type = faiss.METRIC_L2
        if self.metric in ['cosine', 'inner_product']:
            self.logger.warning(
                f'{self.metric} will be output as distance instead of similarity.'
            )
            metric_type = faiss.METRIC_INNER_PRODUCT

        if self.metric not in {'euclidean', 'cosine', 'inner_product'}:
            self.logger.warning(
                'Invalid distance metric for Faiss index construction. Defaulting '
                'to euclidean distance'
            )
        return metric_type

    def is_deleted(self, idx):
        return self._is_deleted[idx]

    def _append_vecs_and_ids(self, doc_ids: List[str], vecs: np.ndarray):
        assert len(doc_ids) == vecs.shape[0]
        if self._faiss_index is None:
            self._init_faiss_index(
                vecs.shape[-1], trained_index_file=self.trained_index_file
            )

        for doc_id in doc_ids:
            self._doc_id_to_offset[doc_id] = len(self._doc_ids)
            self._doc_ids.append(doc_id)
            self._is_deleted.append(0)
        self._index(vecs)

    def _add_delta(self, delta: GENERATOR_DELTA):
        """
        Adding the delta data to the indexer
        :param delta: a generator yielding (id, np.ndarray, last_updated)
        """
        if delta is None:
            self.logger.warning('No data received in Faiss._add_delta. Skipping...')
            return

        for doc_id, vec_array, doc_timestamp in delta:
            self._update_timestamp(doc_timestamp)

            idx = self._doc_id_to_offset.get(doc_id)
            if idx is None:  # add new item
                if vec_array is None:
                    continue
                # shape [1, D]
                vec = vec_array.reshape(1, -1).astype(np.float32)

                self._append_vecs_and_ids([doc_id], vec)
            elif vec_array is None:  # soft delete
                self._is_deleted[idx] = 1
            else:  # update
                # first soft delete
                self._is_deleted[idx] = 1

                # then add the updated doc
                # shape [1, D]
                vec = vec_array.reshape(1, -1).astype(np.float32)
                self._append_vecs_and_ids([doc_id], vec)

    def _update_timestamp(self, doc_timestamp):
        if doc_timestamp:
            if self.last_timestamp.tzname() is None:
                self.last_timestamp = self.last_timestamp.replace(
                    tzinfo=doc_timestamp.tzinfo
                )

            if doc_timestamp > self.last_timestamp:
                self.last_timestamp = doc_timestamp
