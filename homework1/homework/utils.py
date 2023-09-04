import csv

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=None):

        # parsing labels using python csv library
        labels = []

        labels_csv_path = dataset_path + '/labels.csv'
        with open(labels_csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # bypassing the header row
            for row in csv_reader:
                file_name, label = row
                labels.append((file_name, int(label)))

        # initializing variables
        self.dataset_path = dataset_path
        self.transform = transform
        self.labels = labels

        # raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        return len(self.labels)

        # raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        # image is torch.Tensor of size (3,64,64) with range [0,1]
        img_path = self.dataset_path + '/images/' + self.labels[idx][0]
        image = Image.open(img_path)

        # label is int
        label = self.labels[idx][1]

        # return tuple of image, label
        if self.transform:
            image = self.transform(image)

        return image, label

        # raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
