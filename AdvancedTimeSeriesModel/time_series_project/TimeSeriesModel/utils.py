import numpy as np

torch


def generate_synthetic_data(seq_length=50, num_samples=1000):
    X = np.random.rand(num_samples, seq_length, 1)
    Y = np.sum(X, axis=1) + np.random.normal(0, 0.1, (num_samples, 1))
    return X, Y


def create_dataloaders(X, Y, batch_size=32, split_ratio=0.8):
    dataset_size = len(X)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))

    train_indices, val_indices = indices[:split], indices[split:]

    # Creating data samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(
        train_indices
    )
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # Creating data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=list(zip(X, Y)),
        batch_size=batch_size,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=list(zip(X, Y)),
        batch_size=batch_size,
        sampler=val_sampler,
    )

    return train_loader, val_loader
