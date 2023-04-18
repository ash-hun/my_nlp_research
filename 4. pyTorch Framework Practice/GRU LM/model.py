from torch import nn

class GRULanguageModel(nn.Module):
    def __init__(self, hidden_size=30, output_size=10):
        super(GRULanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))

        return output, hidden