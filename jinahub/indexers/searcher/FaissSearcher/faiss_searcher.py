__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import os
import pickle
from datetime import datetime
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from bidict import bidict
from jina import Document, DocumentArray, Executor, requests
from jina.helper import batch_iterator
from jina.logging.logger import JinaLogger
from jina_commons.indexers.dump import import_vectors

GENERATOR_DELTA = Generator[
    Tuple[str, Optional[np.ndarray], Optional[datetime], Optional[bool]], None, None
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
        index_key: str = 'Flat',
        metric: str = 'cosine',
        limit: int = 10,
        nprobe: int = 1,
        ef_construction: int = 80,
        ef_query: int = 20,
        trained_index_file: Optional[str] = None,
        max_num_training_points: Optional[int] = None,
        dump_path: Optional[str] = None,
        prefetch_size: Optional[int] = 512,
        index_traversal_paths: List[str] = ['r'],
        search_traversal_paths: List[str] = ['r'],
        is_distance: bool = True,
        on_gpu: bool = False,
        *args,
        **kwargs,
    ):
        """
        :param index_key: index type supported
            by ``faiss.index_factory``
        :param metric: 'euclidean', 'cosine' or 'inner_product' accepted. Determines which distances to
            optimize by FAISS. euclidean...smaller is better, cosine...larger is better
        :param limit: Number of results to get for each query document in search
        :param nprobe: Number of clusters to consider at search time.
        :param ef_construction: The construction time/accuracy trade-off
        :param ef_query: The query time accuracy/speed trade-off
        :param trained_index_file: the index file dumped from a trained
            index, e.g., ``faiss.index``. If none is provided, `indexed` data will be used
            to train the Indexer (In that case, one must be careful when sharding
            is enabled, because every shard will be trained with its own part of data).
        :param max_num_training_points: Optional argument to consider only a subset of
        data points which will be selected randomly from the available points
        :param dump_path: The path to the directory from where to load, and where to
            save the index state
        :param prefetch_size: the number of data to pre-load into RAM
        :param traversal_paths: The default traverseal path on docs (used for indexing,
            search and update), e.g. ['r'], ['c']
        :param is_distance: Boolean flag that describes if distance metric need to be
            reinterpreted as similarities.
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        self.last_timestamp = datetime.min

        self.num_dim = 0
        self.index_key = index_key
        self.trained_index_file = trained_index_file
        self.max_num_training_points = max_num_training_points

        self.prefetch_size = prefetch_size
        self.metric = metric
        self.limit = limit

        self.normalize = False
        if self.metric == 'cosine':
            self.normalize = True

        self.nprobe = nprobe
        self.ef_construction = ef_construction
        self.ef_query = ef_query

        self.on_gpu = on_gpu

        self.index_traversal_paths = index_traversal_paths
        self.search_traversal_paths = search_traversal_paths
        self.is_distance = is_distance

        self._ids_to_inds = bidict()

        self._is_deleted = set()
        self._prefetch_data = []
        self._faiss_index = None

        dump_path = dump_path or kwargs.get('runtime_args', {}).get('dump_path')
        if dump_path:
            self.load_from_dumps(dump_path, prefetch_size, **kwargs)
        else:
            self.load(self.workspace)

    def load_from_dumps(self, dump_path, prefetch_size, **kwargs):
        if dump_path is not None:
            self.logger.info(
                f'Start building "FaissIndexer" from dump data {dump_path}'
            )
            ids_iter, vecs_iter = import_vectors(
                dump_path, str(self.runtime_args.pea_id)
            )
            iterator = zip(ids_iter, vecs_iter)

            if iterator is not None:
                self.load_from_iterator(iterator, prefetch_size, **kwargs)
        else:
            self.logger.warning(
                'No "dump_path" or "dump_func" passed to "FaissIndexer".'
                ' Use .rolling_update() to re-initialize it...'
            )
            return

    def load_from_iterator(self, iterator, prefetch_size, **kwargs):
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

        _num_dim = self._prefetch_data[0][1].shape[0]
        if self.num_dim == 0:
            self.num_dim = _num_dim

        if self.num_dim != _num_dim:
            raise ValueError(
                'The document should have the same '
                'dimension of embedding as the index, {} != {}'.format(
                    self.num_dim, _num_dim
                )
            )

        self.dtype = self._prefetch_data[0][1].dtype

        self._build_index(iterator)

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

        if hasattr(index, 'hnsw'):
            index.hnsw.efSearch = self.ef_query
            index.hnsw.efConstruction = self.ef_construction

        if not hasattr(index, 'id_map'):
            index = faiss.IndexIDMap2(index)

        self._faiss_index = self.to_device(index)

        self._faiss_index.nprobe = self.nprobe

    def _build_index(self, data_iter: Iterable[Tuple[str, 'np.ndarray']]):
        """Build an advanced index structure from a numpy array.

        :param data_iter: iterator of numpy array containing the vectors to index
        """

        if self._faiss_index is None:
            self._init_faiss_index(
                self.num_dim, trained_index_file=self.trained_index_file
            )

        if not self._faiss_index.is_trained:
            self.logger.info('Taking indexed data as training points...')
            if self.max_num_training_points is None:
                self._prefetch_data.extend(list(data_iter))
            else:
                self.logger.info('Taking indexed data as training points')
                while (
                    self.max_num_training_points
                    and len(self._prefetch_data) < self.max_num_training_points
                ):
                    try:
                        self._prefetch_data.append(next(data_iter))
                    except Exception as _:  # noqa: F841
                        break

            if len(self._prefetch_data) == 0:
                return

            train_data = np.stack([d[1] for d in self._prefetch_data])
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
        self._build_partial_index(data_iter)

    def _build_partial_index(self, data_iter: Iterable[Tuple[str, 'np.ndarray']]):
        if len(self._prefetch_data) > 0:
            embeddings = []
            doc_ids = []
            for d in self._prefetch_data:
                doc_ids.append(d[0])
                embeddings.append(d[1])

                if len(d) > 2 and d[2] is not None:
                    self._update_timestamp(d[2])

            embeddings = np.stack(embeddings).astype(np.float32)
            self._append_vecs_and_ids(embeddings, doc_ids)

            self._prefetch_data.clear()

        for batch_data in batch_iterator(data_iter, self.prefetch_size):
            batch_data = list(batch_data)
            if len(batch_data) == 0:
                break

            embeddings = []
            doc_ids = []
            for d in batch_data:
                if d[1] is None:
                    continue
                doc_ids.append(d[0])
                embeddings.append(d[1])
                if len(d) > 2 and d[2] is not None:
                    self._update_timestamp(d[2])

            embeddings = np.stack(embeddings).astype(np.float32)
            self._append_vecs_and_ids(embeddings, doc_ids)

    @property
    def is_trained(self):
        return self._faiss_index.is_trained if self._faiss_index else False

    @requests(on='/search')
    def search(
        self,
        docs: Optional[DocumentArray],
        parameters: Optional[Dict] = dict(),
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
        if self._faiss_index is None:
            self.logger.warning('Querying against an empty Index')
            return

        limit = int(parameters.get('limit', self.limit))
        traversal_paths = parameters.get('traversal_paths', self.search_traversal_paths)

        # expand topk number guarantee to return topk results
        # TODO WARNING: maybe this would degrade the query speed
        expand_topk = limit + self.deleted_count

        query_docs = docs.traverse_flat(traversal_paths)
        vecs = query_docs.embeddings.astype(np.float32)

        if self.normalize:
            faiss.normalize_L2(vecs)

        dists, ids = self._faiss_index.search(vecs, expand_topk)

        if self.metric in ['cosine', 'inner_product']:
            dists = 1 - dists

        for doc_idx, matches in enumerate(zip(ids, dists)):
            count = 0
            for m_info in zip(*matches):
                idx, dist = m_info

                # this is related with the issue of faiss
                if self._faiss_index.ntotal == 0 or self.is_deleted(idx):
                    continue

                doc_id = self._ids_to_inds.inverse[idx]
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
                if count >= limit:
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

        if self._faiss_index is None:
            self.num_dim = docs.embeddings.shape[-1]
            self._init_faiss_index(
                self.num_dim, trained_index_file=self.trained_index_file
            )

        traversal_paths = parameters.get('traversal_paths', self.index_traversal_paths)
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        try:
            doc_ids = flat_docs.get_attributes('id')
            vecs = flat_docs.embeddings
            self._append_vecs_and_ids(vecs, doc_ids)
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
            pickle.dump(self._ids_to_inds, fp)

        with open(os.path.join(target_path, DELETE_MARKS_FILENAME), "wb") as fp:
            pickle.dump(self._is_deleted, fp)

    def _faiss_index_exist(self, folder_path: str):
        index_path = os.path.join(folder_path, FAISS_INDEX_FILENAME)
        return os.path.exists(index_path)

    def load(self, from_path: Optional[str] = None):
        from_path = from_path if from_path else self.workspace
        self.logger.info(f'Try to restore indexer from {from_path}...')
        try:
            with open(os.path.join(from_path, DOC_IDS_FILENAME), 'rb') as fp:
                self._ids_to_inds = pickle.load(fp)

            with open(os.path.join(from_path, DELETE_MARKS_FILENAME), 'rb') as fp:
                self._is_deleted = pickle.load(fp)

            index = faiss.read_index(os.path.join(from_path, FAISS_INDEX_FILENAME))
            assert index.metric_type == self.metric_type
            assert index.is_trained
            self.num_dim = index.d

            if hasattr(index, 'hnsw'):
                index.hnsw.efSearch = self.ef_query
                index.hnsw.efConstruction = self.ef_construction

            if not hasattr(index, 'id_map'):
                index = faiss.IndexIDMap2(index)

            self._faiss_index = self.to_device(index)
            self._faiss_index.nprobe = self.nprobe

        except FileNotFoundError:
            self.logger.warning(
                'None snapshot is found, you should build the indexer from scratch'
            )
            return False
        except Exception as ex:
            self.logger.warning(f'Exception: {ex}')
            return False

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

        traversal_paths = parameters.get('traversal_paths', self.index_traversal_paths)
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

    def save_trained_model(self, target_path: str):
        if self._faiss_index and self._faiss_index.is_trained:
            faiss.write_index(self._faiss_index, target_path)
            return True
        else:
            self.logger.error('The index instance is not initialized or not trained')
            return False

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: Optional[DocumentArray], **kwargs):
        if docs is None:
            return
        for doc in docs:
            if doc.id in self._ids_to_inds:
                try:
                    reconstruct_embedding = self._faiss_index.reconstruct(
                        self._ids_to_inds[doc.id]
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

    @requests(on='/status')
    def status(self, **kwargs) -> DocumentArray:
        """Return the document containing status information about the indexer.

        The status will contain information on the total number of indexed and deleted
        documents, and on the number of (searchable) documents currently in the index.
        """

        status = Document(
            tags={
                'count_active': self.size,
                'count_indexed': self._faiss_index.ntotal,
                'count_deleted': len(self._is_deleted),
            }
        )
        return DocumentArray([status])

    @requests(on='/clear')
    def clear(self, **kwargs):
        if self._faiss_index is not None:
            self._faiss_index.reset()
            self._ids_to_inds.clear()
            self._is_deleted.clear()

    @property
    def size(self):
        """Return the nr of elements in the index"""
        return self._faiss_index.ntotal - self.deleted_count if self._faiss_index else 0

    @property
    def deleted_count(self):
        return len(self._is_deleted)

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
        return idx in self._is_deleted

    def _append_vecs_and_ids(self, vecs: np.ndarray, doc_ids: List[str]):
        assert len(doc_ids) == vecs.shape[0]
        size = 0
        if len(self._ids_to_inds) > 0:
            size = max(list(self._ids_to_inds.values())) + 1
        indices = []
        for i, doc_id in enumerate(doc_ids):
            idx = size + i
            indices.append(idx)
            if doc_id in self._ids_to_inds:
                self._is_deleted.add(self._ids_to_inds[doc_id])

            self._ids_to_inds.update({doc_id: idx})
        indices = np.array(indices, dtype=np.int64)
        self._faiss_index.add_with_ids(vecs, indices)

    def add_delta_updates(self, delta: GENERATOR_DELTA):
        """
        Adding the delta data to the indexer
        :param delta: a generator yielding (id, np.ndarray, last_updated)
        """
        if delta is None:
            self.logger.warning(
                'No data received in FaissSearcher.add_deleta_updates. Skipping...'
            )
            return

        for batch_data in batch_iterator(delta, self.prefetch_size):
            updated_ids = []
            updated_embeds = []
            updated_idx = []

            batch_data = list(batch_data)
            if len(batch_data) == 0:
                break

            for doc_id, vec, doc_timestamp, is_deleted in batch_data:
                self._update_timestamp(doc_timestamp)
                idx = self._ids_to_inds.get(doc_id, None)

                if idx is None:  # add new item
                    if is_deleted or (vec is None):
                        continue
                    self._append_vecs_and_ids(vec, [doc_id])

                else:
                    updated_idx.append(idx)

                    if (not is_deleted) and (vec is not None):
                        updated_ids.append(doc_id)
                        updated_embeds.append(vec)

            if len(updated_idx) > 0:
                try:
                    self._faiss_index.remove_ids(np.array(updated_idx))
                    for _doc_id in updated_ids:
                        del self._ids_to_inds[_doc_id]
                except Exception as ex:
                    self.logger.warning(f'{ex}')
                    for _idx in updated_idx:
                        self._is_deleted.add(_idx)

            if len(updated_ids) > 0:
                embeddings = np.stack(updated_embeds).astype(np.float32)
                self._append_vecs_and_ids(embeddings, updated_ids)

    def _update_timestamp(self, doc_timestamp):
        if doc_timestamp:
            if self.last_timestamp.tzname() is None:
                self.last_timestamp = self.last_timestamp.replace(
                    tzinfo=doc_timestamp.tzinfo
                )

            if doc_timestamp > self.last_timestamp:
                self.last_timestamp = doc_timestamp
