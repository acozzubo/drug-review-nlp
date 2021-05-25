import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BasicLSTM(nn.Module):
    # This model is based on an LSTM found in Ben Travett's blog
    # https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, padding_idx, embedding_vectors):

        super(BasicLSTM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_vectors, freeze=False, padding_idx=padding_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

        self.hidden_layer = nn.Linear(hidden_dim * 2, output_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))

        packed_embedded = pack_padded_sequence(embedded, text_lengths.to(
            'cpu'), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output)

        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.hidden_layer(hidden)


class BasicAssLSTM_Plus(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, padding_idx, embedding_vectors):
        """
        if extra data is being used in the lsmt, add the length of that 
        """

        super(BasicAssLSTM_Plus, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_vectors, freeze=False, padding_idx=padding_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

        self.hidden_layer = nn.Linear(hidden_dim * 2, output_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths, extra_inputs):
        """
        extra inputs are concatenated along last dimension
        """

        embs = self.embedding(text)
        full_data = torch.cat((embs, extra_inputs), dim=-1)
        remainder = self.dropout(full_data)

        packed_embedded = pack_padded_sequence(remainder, text_lengths.to(
            'cpu'), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        output, output_lengths = pad_packed_sequence(packed_output)

        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.hidden_layer(hidden)
