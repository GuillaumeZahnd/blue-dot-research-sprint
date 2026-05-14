from torch.utils.data import Sampler
import random


class CustomSampler(Sampler):
    def __init__(self, harmful_indices, harmless_indices, batch_size):
        """
        Guarantees exactly batch_size/2 harmful and batch_size/2 harmless samples.
        """
        self.harmful_indices = list(harmful_indices)
        self.harmless_indices = list(harmless_indices)
        self.batch_size = batch_size
        self.num_per_class = batch_size // 2

        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even to maintain 50/50 split.")

        # Total number of full 50/50 batches possible
        self.num_batches = min(len(self.harmful_indices), len(self.harmless_indices)) // self.num_per_class

    def __iter__(self):
        # Shuffle both pools independently each epoch
        random.shuffle(self.harmful_indices)
        random.shuffle(self.harmless_indices)

        indices = []
        for i in range(self.num_batches):
            # Deterministically slice the pools
            h_chunk = self.harmful_indices[i * self.num_per_class : (i + 1) * self.num_per_class]
            hl_chunk = self.harmless_indices[i * self.num_per_class : (i + 1) * self.num_per_class]

            combined = h_chunk + hl_chunk
            # Shuffle internal batch order so harmful isn't always at the start of the 4
            random.shuffle(combined)
            indices.extend(combined)

        return iter(indices)

    def __len__(self):
        return self.num_batches * self.batch_size
