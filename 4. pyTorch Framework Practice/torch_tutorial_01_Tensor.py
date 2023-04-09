import torch
import numpy as np
from torch import nn

# 데이터로부터 직접(directly) 생성하기
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

# Numpy 배열로부터 생성하기
l = torch.ones(5)
np_array = np.array(data)
x_np = torch.from_numpy(np_array) # Numpy to Tensor
tt = l.numpy() # Tensor to Numpy

x_ones = torch.ones_like(x_data) # x_data의 속성을 유지한다.
print(f"Ones Tensor : {x_ones}")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어쓴다.
print(f"Random Tensor : {x_rand}")

# 무작위(random) 또는 상수(constant) 값을 사용하기
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# GPU가 존재하면 텐서를 이동합니다
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print("cuda is available")
else:
    print("cuda is not available")

# Numpy식의 표준 인덱싱과 슬라이싱
ttt = torch.ones(4,4) # 0벡터 선언
print(f"First row : {ttt[0]}")
print(f"First Column : {ttt[:, 0]}")
print(f"Las Column : {ttt[..., -1]}")
ttt[:, 1] = 0
print(ttt)

# torch.cat을 사용하여 주어진 차원에 따라 일련의 텐서를 연결할 수 있다.
t1 = torch.cat([ttt, ttt, ttt], dim=1)
print(t1)

# ===========================================================================================
# Arithmetic Operations
# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# ===========================================================================================
# Single-Element Tensor
# 텐서의 모든 값을 하나로 집계(aggregate)하여 요소가 하나인 텐서의 경우, item() 을 사용하여 Python 숫자 값으로 변환할 수 있습니다
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# ===========================================================================================
# In-Place Operations
# 연산 결과를 피연산자(operand)에 저장하는 연산을 바꿔치기 연산이라고 부르며, _ 접미사를 갖습니다.
# 예를 들어: x.copy_(y) 나 x.t_() 는 x 를 변경합니다.
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
# 바꿔치기 연산은 메모리를 일부 절약하지만, 기록(history)이 즉시 삭제되어 도함수(derivative) 계산에 문제가 발생할 수 있습니다. 따라서, 사용을 권장하지 않습니다.