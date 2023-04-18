import torch
import numpy as np

def train(inputs, labels, model, criterion, optimizer, max_grad_norm=None):
    hidden_size = model.hidden_size
    batch_size = inputs.size()[0]
    hidden = torch.zeros((1, batch_size, hidden_size))
    input_length = inputs.size()[1]
    loss = 0

    teacher_forcing = True if np.random.random() < 0.5 else False
    lm_inputs = inputs[:, 0].unsquezze(-1)
    for i in range(input_length):
        output, hidden = model(lm_inputs, hidden)
        output = output.squeeze(1)
        loss += criterion(output, labels[:, i])

        if teacher_forcing:
            lm_inputs = labels[:, i].unsqueeze(-1)
        else:
            topv, topi = output.topk(1)
            lm_inputs = topi
        
    loss.backward()
    if max_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss