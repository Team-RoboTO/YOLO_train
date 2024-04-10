from torch.utils.data import Dataset
from PIL import Image
import os, csv

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []

        with open(annotations_file, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.img_labels.append(row)


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = Image.open(image_path)
        label = int(self.img_labels[idx][1])

        if self.transform:
            image = self.transform(image)

        return image, label