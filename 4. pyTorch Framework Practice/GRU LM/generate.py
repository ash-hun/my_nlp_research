import torch

def generate_sentence_from_bos(model, vocab, bos=1):
    indice = [bos]
    hidden = torch.zeros((1, 1, model.hidden_size))
    lm_inputs = torch.tensor(indice).unsqueeze(-1)
    i2v = {v:k for k, v in vocab.items()}

    cnt = 0
    eos = vocab['</s>']
    generated_sequence = [lm_inputs[0].data.item()]
    while True:
        if cnt == 30:
            break
        output, hidden = model(lm_inputs, hidden)
        output = output.squeeze(1)
        topv, topi = output.topk(1)
        lm_inputs = topi
        if topi.data.item() == eos:
            tokens = list(map(lambda w : i2v[w], generated_sequence))
            generated_sequence = ' '.join(tokens)
            return generated_sequence
        
        generated_sequence.append(topi.data.item())
        cnt += 1
    print('max iteration reached. therfore finishing forcefully')
    tokens = list(map(lambda w: i2v[w], generated_sequence))

    generated_sequence = ' '.join(tokens)
    return generated_sequence