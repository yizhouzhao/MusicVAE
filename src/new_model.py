from params import *

class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_sizes,
                 has_bias=True,
                 dropout_rate=0.5,
                 activate_final=False):
        super(MLP, self).__init__()

        dims = [input_size
                ] + [output_sizes[i] for i in range(len(output_sizes))]

        self._has_bias = has_bias
        self._activate_final = activate_final
        self._dropout_rate = dropout_rate

        self._linear = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=has_bias)
            for i in range(len(dims) - 1)
        ])

    def forward(self, x, training=True):
        for _, layer in enumerate(self._linear):
            x = layer(x)
        if self._dropout_rate not in (None, 0) and training:
            x = nn.Dropout(p=self._dropout_rate)(x)
        if self._activate_final:
            x = nn.ReLU(x)
        return x

class VAECell(nn.Module):
    def __init__(self, latent_features):
        super(VAECell, self).__init__()
        self.latent_features = latent_features

        # data goes into bidirectional encoder
        self.encoder = torch.nn.LSTM(
            batch_first=True,
            input_size=input_size,
            hidden_size=enc_hidden_size,
            num_layers=1,
            bidirectional=True)

        # encoded data goes onto connect linear layer. inputs must be*2 because LSTM is bidirectional
        # output must be 2*latentspace because it needs to be split into miu and sigma right after.
        self.encoderOut = MLP(enc_hidden_size * 2, [400, latent_features * 2 * m_key_count], dropout_rate=dropout_rate)

    # used to initialize the hidden layer of the encoder to zero before every batch
    def init_hidden(self, batch_size):
        # must be 2 x batch x hidden_size because its a bi-directional LSTM
        init = torch.zeros(2, batch_size, enc_hidden_size, device=device)
        c0 = torch.zeros(2, batch_size, enc_hidden_size, device=device)

        # 2 because has 2 layers
        # n_layers_conductor
        init_conductor = torch.zeros(1, batch_size, decoders_initial_size, device=device)
        c_condunctor = torch.zeros(1, batch_size, decoders_initial_size, device=device)

        return init, c0, init_conductor, c_condunctor


    def forward(self, x):
        #add empty sound in the beginning
        batch_size = x.size(0)
        note = torch.zeros(batch_size, 1, NUM_PITCHES, device=device)
        the_input = torch.cat([note, x], dim=1)

        # creates hidden layer values
        h0, c0, hconductor, cconductor = self.init_hidden(batch_size)

        # resets encoder at the beginning of every batch and gives it x
        x, hidden = self.encoder(x, (h0, c0))
        x = self.encoderOut(x)

        mu_var_list = torch.chunk(x, 2 * m_key_count, dim=-1)

