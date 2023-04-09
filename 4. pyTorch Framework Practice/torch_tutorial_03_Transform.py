import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="./data03",
    train=True,
    download=True,
    # ToTensor() -> PIL Image나 Numpy를 정해진 타입의 Tensor 형태로 변환하고 이미지 픽셀의 크기값을 [0.,1.] 범위로 비례하여 조정한다.
    transform=ToTensor(),
    # Lambda Transform -> 사용자 정의 람다함수를 적용한다. 
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
