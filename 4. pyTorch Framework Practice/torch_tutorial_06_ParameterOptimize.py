import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


training_data = datasets.FashionMNIST(
    root="./data06",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="./data06",
    train=False,
    download=True,
    transform=ToTensor()
)

TrainData = training_data
TestData = test_data

train_dataloader = DataLoader(TrainData, batch_size=64)
test_dataloader = DataLoader(TestData, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# ---------------------------------------------------------------------------
# Hyperparameter
# 하이퍼 파라미터는 모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수이다.
# 학습 시 아래와 같은 하이퍼 파라미터를 정의한다.
# ---------------------------------------------------------------------------
# 1. Epoch : 데이터 셋을 반복하는 횟수
epochs = 5
# 2. Batch Size : 매개 변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
lr = 1e-3
# 3. Learning Rate : 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 값이 작을수록 학습속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있다.
batch_size = 64

# ---------------------------------------------------------------------------
# Optimization Loop
# 최적화 단계를 통해 모델을 학습하고 최적화할 수 있다. 최적화 단계의 각 반복(iteration)을 에폭이라고 부른다.
# 하나의 에폭은 다음 두 부분으로 구성된다.
# 1. 학습단계 (train loop) - 학습용 데이터셋을 반복(iterate)하고 최적의 매개변수로 수렴
# 2. 검증/테스트 단계 (validation/test loop) - 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터 셋을 반복한다.

# 손실 함수를 초기화합니다.
loss_fn = nn.CrossEntropyLoss()

# 학습 단계(loop)에서 최적화는 세단계로 이뤄집니다:
# optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를 재설정합니다. 기본적으로 변화도는 더해지기(add up) 때문에 중복 계산을 막기 위해 반복할 때마다 명시적으로 0으로 설정합니다.
# loss.backwards()를 호출하여 예측 손실(prediction loss)을 역전파합니다. PyTorch는 각 매개변수에 대한 손실의 변화도를 저장합니다.
# 변화도를 계산한 뒤에는 optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정합니다.
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")