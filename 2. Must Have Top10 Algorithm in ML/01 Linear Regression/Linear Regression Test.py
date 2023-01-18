import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 현재 파이썬 코드를 재실행해도 다음에 같은 결과가 나오도록 랜덤 시드(random seed)를 준다.
torch.manual_seed(1)

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


optimizer = optim.SGD([w, b], lr=0.01)


nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = w * x_train + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train)**2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w.item(), b.item(), cost.item()
        ))
