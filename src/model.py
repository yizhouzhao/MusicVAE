from src.params import *


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_features, teacher_forcing, eps_i):
        super(VariationalAutoencoder, self).__init__()

        self.teacher_forcing = teacher_forcing
        self.eps_i = eps_i

        self.latent_features = latent_features

        # data goes into bidirectional encoder
        self.encoder = torch.nn.LSTM(batch_first=True,
                                     input_size=input_size,
                                     hidden_size=enc_hidden_size,
                                     num_layers=1,
                                     bidirectional=True)

        # encoded data goes onto connect linear layer. inputs must be*2 because LSTM is bidirectional
        # output must be 2*latentspace because it needs to be split into miu and sigma right after.
        self.encoderOut = nn.Linear(in_features=enc_hidden_size * 2,
                                    out_features=latent_features * 2)

        # after being converted data goes through a fully connected layer
        self.linear_z = nn.Linear(in_features=latent_features,
                                  out_features=decoders_initial_size)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.worddropout = nn.Dropout2d(p=dropout_rate)

        # Define the conductor and note decoder
        self.conductor = nn.LSTM(decoders_initial_size,
                                 decoders_initial_size,
                                 num_layers=1,
                                 batch_first=True)
        self.decoder = nn.LSTM(NUM_PITCHES + decoders_initial_size,
                               decoders_initial_size,
                               num_layers=1,
                               batch_first=True)

        # Linear note to note type (classes/pitches)
        self.linear = nn.Linear(decoders_initial_size, NUM_PITCHES)

    # used to initialize the hidden layer of the encoder to zero before every batch
    # nn.LSTM will do this by itself, this might be redunt. Look at: https://discuss.pytorch.org/t/when-to-initialize-lstm-hidden-state/2323/16
    def init_hidden(self, batch_size):
        # must be 2 x batch x hidden_size because its a bi-directional LSTM
        init = torch.zeros(2, batch_size, enc_hidden_size, device=device)
        c0 = torch.zeros(2, batch_size, enc_hidden_size, device=device)

        # 2 because has 2 layers
        # n_layers_conductor
        init_conductor = torch.zeros(1,
                                     batch_size,
                                     decoders_initial_size,
                                     device=device)
        c_condunctor = torch.zeros(1,
                                   batch_size,
                                   decoders_initial_size,
                                   device=device)

        return init, c0, init_conductor, c_condunctor

    # Coin toss to determine whether to use teacher forcing on a note(Scheduled sampling)
    # Will always be True for eps_i = 1.
    def use_teacher_forcing(self):
        with torch.no_grad():
            tf = np.random.rand(1)[0] <= self.eps_i
        return tf

    def set_scheduled_sampling(self, eps_i):
        self.eps_i = 1  #eps_i

    def forward(self, x):
        batch_size = x.size(0)

        note = torch.zeros(batch_size, 1, NUM_PITCHES, device=device)

        the_input = torch.cat([note, x], dim=1)

        outputs = {}

        # creates hidden layer values
        # nn.LSTM() will create zero initial states by default
        h0, c0, hconductor, cconductor = self.init_hidden(batch_size)

        x = self.worddropout(x)

        # resets encoder at the beginning of every batch and gives it x
        x, _ = self.encoder(x, (h0, c0))

        # x=self.dropout(x)

        # goes from 4096 to 1024
        x = self.encoderOut(x)

        # x=self.dropout(x)

        # Split encoder outputs into a mean and variance vector
        mu, log_var = torch.chunk(x, 2, dim=-1)

        # Make sure that the log variance is positive
        log_var = softplus(log_var)

        # :- Reparametrisation trick
        # a sample from N(mu, sigma) is mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
        # TODO: try gumbel-softmax for easier interpretion?

        # Don't propagate gradients through randomness
        with torch.no_grad():
            batch_size = mu.size(0)
            epsilon = torch.randn(batch_size, 1, self.latent_features)

            if use_cuda:
                epsilon = epsilon.cuda()

        # setting sigma
        sigma = torch.exp(log_var * 2)

        # generate z - latent space
        z = mu + epsilon * sigma

        # decrese space
        z = self.linear_z(z)

        # z=self.dropout(z)

        # make dimensions fit (NOT SURE IF THIS IS ENTIRELY CORRECT)
        # z = z.permute(1,0,2)

        # DECODER ##############

        conductor_hidden = (hconductor, cconductor)

        counter = 0

        notes = torch.zeros(batch_size,
                            TOTAL_NOTES,
                            NUM_PITCHES,
                            device=device)

        # For the first timestep the note is the embedding
        # This line is duplicate of line 79
        note = torch.zeros(batch_size, 1, NUM_PITCHES, device=device)

        # print(z[:,0,:].view(batch_size,1, -1).shape)

        # Go through each element in the latent sequence
        # TODO: z.shape[1] = 512? why for i in range(16), only use the first 16?
        for i in range(16):
            embedding, conductor_hidden = self.conductor(
                z[:, 16 * i, :].view(batch_size, 1, -1), conductor_hidden)

            if self.use_teacher_forcing():

                # Reset the decoder state of each 16 bar sequence
                decoder_hidden = (torch.randn(1,
                                              batch_size,
                                              decoders_initial_size,
                                              device=device),
                                  torch.randn(1,
                                              batch_size,
                                              decoders_initial_size,
                                              device=device))

                embedding = embedding.expand(batch_size, NOTESPERBAR,
                                             embedding.shape[2])

                e = torch.cat(
                    [embedding, the_input[:, range(i * 16, i * 16 + 16), :]],
                    dim=-1)

                notes2, decoder_hidden = self.decoder(e, decoder_hidden)

                aux = self.linear(notes2)
                aux = torch.softmax(aux, dim=2)

                # generates 16 notes per batch at a time
                notes[:, range(i * 16, i * 16 + 16), :] = aux
            else:
                # Reset the decoder state of each 16 bar sequence
                decoder_hidden = (torch.randn(1,
                                              batch_size,
                                              decoders_initial_size,
                                              device=device),
                                  torch.randn(1,
                                              batch_size,
                                              decoders_initial_size,
                                              device=device))

                for _ in range(sequence_length):
                    # Concat embedding with previous note

                    e = torch.cat([embedding, note], dim=-1)
                    e = e.view(batch_size, 1, -1)

                    # Generate a single note (for each batch)
                    note, decoder_hidden = self.decoder(e, decoder_hidden)

                    aux = self.linear(note)
                    aux = torch.sigmoid(aux)

                    notes[:, counter, :] = aux.squeeze()

                    note = aux

                    counter = counter + 1

        outputs["x_hat"] = notes
        outputs["z"] = z
        outputs["mu"] = mu
        outputs["log_var"] = log_var

        return outputs