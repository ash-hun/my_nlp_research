from dataset import GRULanguageModelDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


text = 'she sells sea shells by the sea shore'
dataset = GRULanguageModelDataset(text)
for d in dataset:
    print(d)
    break

print(dataset.vocab)

def collate_fn(batch):
    batch = pad_sequence(batch, batch_first=True)
    return batch

dataloader = DataLoader(dataset, collate_fn = collate_fn, batch_size = 16)

for d in dataloader:
    print(d)
    print(d.shape)
    break
