import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
import os
from torch.utils.data import DataLoader

training = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

testing = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_data = training
test_data = testing

def showMNIST():
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols*rows+1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

class CustomImageDataset(Dataset):#사용자 정의 데이터셋은 반드시 아래 3개 함수를 구현해야 한다.

    # __init__ 함수는 Dataset 객체가 생성될때 한번만 실행된다. 이미지와 주석파일(annotation_file)이 포함된 디렉토리와 두가지 변형을 초기화한다.
    def __init__(self, annotaions_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotaions_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    # __len__ 함수는 데이터셋의 샘플 개수를 반환한다.
    def __len__(self):
        return len(self.img_labels)

    # __getitem__함수는 주어진 인덱스에 해당하는 샘플을 데이터셋에서 불러오고 반환한다.
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__=="__main__":
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()} / {train_features[0]}")
    print(f"Labels batch shape: {train_labels.size()} / {train_labels[0]}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap='gray')
    plt.show()
    print(f"Label: {label}")