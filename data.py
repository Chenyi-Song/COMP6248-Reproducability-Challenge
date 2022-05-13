from torch.utils.data import DataLoader, random_split

# from torchfm.dataset.avazu import AvazuDataset
# from torchfm.dataset.criteo import CriteoDataset
from movielens import MovieLensDataset


def get_dataset(name, path):
    if name == 'movielens':
        return MovieLensDataset(path)
    # elif name == 'criteo':
    #     return CriteoDataset(path)
    # elif name == 'avazu':
    #     return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_dataloader(dataset, batch_size=1024):
    data_len = len(dataset)
    train_length = int(data_len * 0.8)
    valid_length = int(data_len * 0.1)
    test_length = data_len - train_length - valid_length

    train_dataset, valid_dataset, test_dataset = random_split(dataset, (
        train_length, valid_length, test_length))

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, valid_data_loader, test_data_loader