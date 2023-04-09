import torch
import torchvision.models as models


# 모델 가중치 저장하고 불러오기.
# Pytorch 모델은 학습한 매개변수를 state_dict라고 불리는 내부 상태 사전에 저장한다.
# 이 상태값들은 torch.save 메소드를 활용하여 저장할 수 있다.
model = models.vgg16(pretrained=True) # pretrained=True 는 모델의 기본 가중치를 불러온다.
torch.save(model.state_dict(), "model_weights.pth")

# 모델 가중치 불러오기.
model2 = models.vgg16()
model2.load_state_dict(torch.load('model_weights.pth'))
model2.eval() # 모델을 평가모드로 전환한다.
print(model2)

# 모델의 전체 (구조 + 가중치)를 저장하기
torch.save(model, 'vgg16.pth')

# 모델 불러오기.
model3 = torch.load('vgg16.pth')
print(model3)