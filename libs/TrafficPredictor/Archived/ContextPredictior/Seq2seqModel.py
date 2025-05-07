import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        # First layer: input_dim -> hidden_dim*2
        self.lstm_layers.append(
            nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, dropout=dropout)
        )
        self.ln_layers.append(nn.LayerNorm(hidden_dim * 2))
        # For the first layer, we need to project the residual from input_dim to hidden_dim*2
        self.residual_projections.append(nn.Linear(input_dim, hidden_dim * 2))

        # Subsequent layers: hidden_dim*2 -> hidden_dim*2
        for _ in range(n_layers - 1):
            self.lstm_layers.append(nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1, bidirectional=True))
            self.ln_layers.append(nn.LayerNorm(hidden_dim * 2))
            # No projection needed since dimensions match (24 -> 24)
            self.residual_projections.append(None)

    def forward(self, src):
        # src: [src_len, batch_size, input_dim]
        output = src
        hidden_states = []
        cell_states = []

        for i, layer in enumerate(self.lstm_layers):
            residual = output
            output, (hidden, cell) = layer(output)
            output = self.ln_layers[i](output)

            # Residual connection
            if self.residual_projections[i] is not None:
                # Project residual to match output dimension
                residual = self.residual_projections[i](residual)

            output = output + residual

            hidden_states.append(self._combine_directions(hidden))
            cell_states.append(self._combine_directions(cell))

        return output, hidden_states, cell_states

    def _combine_directions(self, tensor):
        # tensor: [2, batch_size, hidden_dim]
        forward = tensor[0:1]  # [1, batch_size, hidden_dim]
        backward = tensor[1:2] # [1, batch_size, hidden_dim]
        return torch.cat((forward, backward), dim=2) # [1, batch_size, hidden_dim * 2]


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim * 2  # Match encoder's output dim
        self.n_layers = n_layers
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        # First layer: output_dim -> hidden_dim*2
        self.lstm_layers.append(nn.LSTM(output_dim, self.hidden_dim, num_layers=1, dropout=dropout))
        self.ln_layers.append(nn.LayerNorm(self.hidden_dim))
        # Project residual from output_dim to hidden_dim*2
        self.residual_projections.append(nn.Linear(output_dim, self.hidden_dim))

        # Subsequent layers: hidden_dim*2 -> hidden_dim*2
        for _ in range(n_layers - 1):
            self.lstm_layers.append(nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1))
            self.ln_layers.append(nn.LayerNorm(self.hidden_dim))
            self.residual_projections.append(None)

        self.fc_out = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input, hidden_states, cell_states):
        # input: [1, batch_size, output_dim]
        output = input
        next_hidden_states = []
        next_cell_states = []

        for i, layer in enumerate(self.lstm_layers):
            residual = output
            hidden = hidden_states[i]
            cell = cell_states[i]

            output, (next_hidden, next_cell) = layer(output, (hidden, cell))
            output = self.ln_layers[i](output)

            # Residual connection
            if self.residual_projections[i] is not None:
                residual = self.residual_projections[i](residual)

            output = output + residual

            next_hidden_states.append(next_hidden)
            next_cell_states.append(next_cell)

        prediction = self.fc_out(output.squeeze(0))
        return prediction, next_hidden_states, next_cell_states


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(output_dim, hidden_dim, n_layers, dropout)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size, input_dim]
        # trg: [trg_len, batch_size, output_dim]
        _, hidden_states, cell_states = self.encoder(src)

        trg_len, batch_size, output_dim = trg.size()
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(src.device)
        #input = src[-1, :, :].unsqueeze(0)  # Start decoding from the last encoder state
        input = torch.zeros(1, batch_size, output_dim).to(src.device)
        for t in range(trg_len):
            output, hidden_states, cell_states = self.decoder(input, hidden_states, cell_states)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t].unsqueeze(0) if teacher_force else output.unsqueeze(0)

        return outputs