import numpy as np

timesteps = 10 # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_size = 4 # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8 # 은닉 상태의 크기. 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_size)) # 입력에 해당하는 2D 텐서
hidden_state_t = np.zeros((hidden_size,)) # 초기 은닉 상태는 0(벡터)로 초기화

Wx = np.random.random((hidden_size, input_size)) # (8,4) 크기의 2D 텐서, 입력에 대한 가중치
Wh = np.random.random((hidden_size, hidden_size)) # (8,8) 크기의 2D 텐서, 은닉상태에 대한 가중치

b = np.random.random((hidden_size,)) # (8, ) 크기의 1D 텐서, 편향(bias)

total_hidden_states = []

# memory cells
for input_t in inputs:
    # Wx * Xt + Wh * H(t-1) + b
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t))
    print(np.shape(total_hidden_states)) # 각 시점 t별 메모리 셀의 출력 크기 : (timestep, outpt_dim)
    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis=0) # 출력 보기좋게!
print(total_hidden_states)