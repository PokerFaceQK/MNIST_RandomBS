from torch.utils.data import Dataset
import torch


class NonFixedBsMnist(Dataset):
    def __init__(self, data_path, bs=None, bs_max=128, transform=None):
        super(NonFixedBsMnist, self).__init__()
        self._data_path = data_path
        self.bs_max = bs_max
        self.bs = bs
        self.fixed_bs = bs is not None
        self._data, self._labels = torch.load(self._data_path)
        self._batch_sizes = self._generate_bs()
        self._offset = torch.zeros_like(self._batch_sizes)
        self._offset[1:] = torch.cumsum(self._batch_sizes[:-1], dim=0)
        self._data, self._labels = self._data.float() / 255, self._labels.long()
        self.transform = transform
        
    
    def _generate_bs(self):
        if self.fixed_bs:
            sizes = [torch.tensor(self.bs)] * (self._data.size(0) // self.bs) + [self._data.size(0) % self.bs]
            return torch.LongTensor(sizes)
        size_sum = 0
        sizes = []
        top = self._data.size(0) - self.bs_max
        while size_sum < top:
            size_tmp = torch.squeeze(torch.randint(1, self.bs_max, (1, )))
            size_sum = size_sum + size_tmp
            sizes.append(size_tmp)
        sizes.append(self._data.size(0) - size_sum)
        return torch.LongTensor(sizes)
        
    def __getitem__(self, idx):
        start, end = self._offset[idx], self._offset[idx] + self._batch_sizes[idx]
        return self.transform(self._data[start:end]), self._labels[start:end]
    
    def __len__(self):
        return self._batch_sizes.size(0)


if __name__ == "__main__":
    trainset = NonFixedBsMnist(data_path='../data/MNIST/processed/training.pt')
    pass
