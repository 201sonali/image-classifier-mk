import csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.transform = transform

        # setting the dataset path
        self.dataset_path = dataset_path

        # parsing labels using csv library
        labels = []
        with open(os.path.join(dataset_path, 'labels.csv'), 'r') as csv_file:
        #with open('../data/train/labels.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            for row in csv_reader:
                file_name, label, null = row
                img_path = os.path.join(dataset_path, file_name)
                #img_path = os.path.join('../data/train/', file_name)
                image = Image.open(img_path)
                image = transforms.ToTensor()(image)

                # number label categories
                if label == "background":
                    label_number = 0
                elif label == "kart":
                    label_number = 1
                elif label == "pickup":
                    label_number = 2
                elif label == "nitro":
                    label_number = 3
                elif label == "bomb":
                    label_number = 4
                else:
                    label_number = 5  # projectile
                labels.append((image, int(label_number)))

        self.labels = labels  # additional self variable

        # raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        return len(self.labels)

        # raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        # image is torch.Tensor of size (3,64,64) with range [0,1]
        image, label = self.labels[idx]

        # transform if it exists
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
