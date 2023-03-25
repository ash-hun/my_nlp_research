import torch
import torch.nn as nn

# hyperparameter
input_size = 5 # 입력의 크기
hidden_size = 8 # 은닉 상태의 크기

# (batch size, time step, input size)
inputs = torch.Tensor(1, 10, 5)

# batch_first = True : 입력 텐서의 첫번째 차원이 배치 크기이다!
cell = nn.RNN(input_size, hidden_size, batch_first=True)

# RNN Cell은 반환값이 2개
# 첫 번째 : 모든 시점(timesteps)의 은닉 상태
# 두 번째 : 마지막 시점(timestep)의 은닉 상태
outputs, _status = cell(inputs)
print(f'outputs : {outputs.shape} / _status : {_status.shape}')

# ----------------------------------------------------------
# Deep RNN

inputs2 = torch.Tensor(1, 10, 5)
cell2  = nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True)
outputs2, _status2 = cell2(inputs2)
print(f'outputs : {outputs2.shape} / _status : {_status2.shape}')

# ----------------------------------------------------------
# Bidirection RNN

inputs_bi = torch.Tensor(1, 10, 5)
cell_bi  = nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True, bidirectional=True)
outputs_bi, _status_bi = cell_bi(inputs_bi)
print(f'outputs : {outputs_bi.shape} / _status : {_status_bi.shape}')
