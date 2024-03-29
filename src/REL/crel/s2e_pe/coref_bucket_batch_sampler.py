import logging
import math
import random
from typing import Iterable, List

from torch.utils import data
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BucketBatchSampler(DataLoader):
    def __init__(
        self,
        data_source: data.Dataset,
        max_total_seq_len: int,
        sorting_keys: List[str] = None,
        padding_noise: float = 0.1,
        drop_last: bool = False,
        batch_size_1: bool = False,
    ):
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.max_total_seq_len = max_total_seq_len
        self.data_source = data_source
        data_source.examples.sort(key=lambda x: len(x[1].token_ids), reverse=True)
        self.drop_last = drop_last
        self.batches = (
            self.prepare_batches() if not batch_size_1 else self.prepare_eval_batches()
        )

    def prepare_batches(self):
        batches = []
        batch = []
        per_example_batch_len = 0
        for (
            doc_key,
            elem,
        ) in self.data_source:  # NOTE: The `doc_key` also contains subtoken map
            batch.append(elem)
            batch = self.data_source.pad_batch(batch, len(batch[0].token_ids))
            batches.append((doc_key, batch))
            batch = []

        if len(batch) == 0:
            return batches
        batch = self.data_source.pad_batch(batch, len(batch[0].token_ids))
        batches.append(
            (doc_key, batch)
        )  # 220229: Change to return dockey as `prepare_eval_batches()` does.
        return batches

    def __iter__(self) -> Iterable[List[int]]:
        random.shuffle(self.batches)
        yield from self.batches

    def __len__(self):
        return len(self.batches)

    def calc_effective_per_example_batch_len(self, example_len):
        return math.ceil((example_len + 2) / 512) * 512

    def prepare_eval_batches(self):
        batches = []
        for doc_key, elem in self.data_source:
            batch = self.data_source.pad_batch([elem], len(elem.token_ids))
            batches.append((doc_key, batch))
        return batches
