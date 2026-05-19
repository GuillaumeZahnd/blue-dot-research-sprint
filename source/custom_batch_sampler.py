from torch.utils.data import Sampler
import random


class CustomBatchSampler(Sampler):
    def __init__(self, harmful_indices, harmless_indices, batch_size):
        """Sampler returning batches that contain 50% harmful and 50% harmless samples"""
        self.harmful_indices = list(harmful_indices)
        self.harmless_indices = list(harmless_indices)
        self.batch_size = batch_size
        self.nb_chunks_per_class = batch_size // 2

        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even to maintain 50/50 split.")

        # Total number of batches, each containing 50% harmful and 50% harmless samples
        self.nb_batches = min(len(self.harmful_indices), len(self.harmless_indices)) // self.nb_chunks_per_class

    def __iter__(self):
        # Shuffle both pools independently each epoch
        random.shuffle(self.harmful_indices)
        random.shuffle(self.harmless_indices)

        indices = []
        for i in range(self.nb_batches):

            # Deterministically slice the pools to retrieve an equal number of samples per category
            harmful_chunk = self.harmful_indices[i * self.nb_chunks_per_class : (i + 1) * self.nb_chunks_per_class]
            harmless_chunk = self.harmless_indices[i * self.nb_chunks_per_class : (i + 1) * self.nb_chunks_per_class]

            batch = harmful_chunk + harmless_chunk
            random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.nb_batches
