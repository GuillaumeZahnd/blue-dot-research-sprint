import torch
from torch.utils.data import Sampler
import random


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.harmful_indices = [i for i, x in enumerate(dataset["is_harmful"]) if x == 1]
        self.harmless_indices = [i for i, x in enumerate(dataset["is_harmful"]) if x == 0]
        self.n_samples = min(len(self.harmful_indices), len(self.harmless_indices))

    def __iter__(self):
        random.shuffle(self.harmful_indices)
        random.shuffle(self.harmless_indices)
        
        samples_per_class = self.batch_size // 2
        for i in range(self.n_samples // samples_per_class):
            batch = []
            start, end = i * samples_per_class, (i + 1) * samples_per_class
            
            batch.extend(self.harmless_indices[start:end])
            batch.extend(self.harmful_indices[start:end])
            
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return (self.n_samples * 2) // self.batch_size
        


class StratifiedBatchSampler(Sampler):
    def __init__(self, harmless_indices, harmful_indices, batch_size):
        self.harmless_indices = harmless_indices
        self.harmful_indices = harmful_indices
        self.batch_size = batch_size
        self.num_batches = min(len(harmless_indices), len(harmful_indices)) // (batch_size // 2)
        
    def __iter__(self):
        # Shuffle indices at the start of every epoch
        random.shuffle(self.harmless_indices)
        random.shuffle(self.harmful_indices)
        
        for i in range(self.num_batches):
            start = i * (self.batch_size // 2)
            end = start + (self.batch_size // 2)
            
            # Pull 50% from each bucket
            batch = self.harmless_indices[start:end] + self.harmful_indices[start:end]
            # Shuffle the batch so harmful samples aren't always at the end
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches        
