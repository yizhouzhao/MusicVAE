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

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        print("Use attention type %s" % method)
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)

        # Create variable to store attention energies
        attn_energies = torch.zeros(this_batch_size,
                                    max_len)  # B x S

        if use_cuda:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[b, :],
                                                 encoder_outputs[b, i])

        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = torch.dot(encoder_output, hidden)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(energy, hidden)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 0))
            energy = self.v.dot(energy)
            return energy

class VAECell(nn.Module):
    def __init__(self, latent_features):
        super(VAECell, self).__init__()
        self.latent_features = latent_features

        #attention: vertical
        self.attn_v = Attn("dot", decoders_initial_size)

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

        # after being converted data goes through a fully connected layer
        self.linear_z = nn.Linear(in_features=latent_features, out_features=decoders_initial_size)

        #conduction to transform z
        #decoders_initial_size + context_size(decoders_initial_size)
        self.conductor = nn.LSTM(decoders_initial_size * 2, decoders_initial_size, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(NUM_PITCHES + decoders_initial_size, decoders_initial_size, num_layers=1,
                               batch_first=True)

        # Linear note to note type (classes/pitches)
        self.linear = nn.Linear(decoders_initial_size, NUM_PITCHES)

    # used to initialize the hidden layer of the encoder to zero before every batch
    def init_hidden(self, batch_size):
        # must be 2 x batch x hidden_size because its a bi-directional LSTM
        init = torch.zeros(2, batch_size, enc_hidden_size, device=device)
        c0 = torch.zeros(2, batch_size, enc_hidden_size, device=device)

        # 2 because has 2 layers
        # n_layers_conductor
        init_conductor = torch.zeros(1, batch_size, m_key_count, decoders_initial_size, device=device)
        c_condunctor = torch.zeros(1, batch_size, m_key_count, decoders_initial_size, device=device)

        h_decoder = torch.zeros(1, batch_size, m_key_count, decoders_initial_size, device=device)
        c_decoder = torch.zeros(1, batch_size, m_key_count, decoders_initial_size, device=device)

        return init, c0, init_conductor, c_condunctor, h_decoder, c_decoder


    def forward(self, x):
        #add empty sound in the beginning
        batch_size = x.size(0)
        note_length = x.size(1)
        note = torch.zeros(batch_size, 1, NUM_PITCHES, device=device)

        #music notes record
        notes = torch.zeros(batch_size, note_length, NUM_PITCHES, device=device)

        #music notes with multi_channel
        multi_notes = torch.zeros(batch_size, note_length, m_key_count, NUM_PITCHES, device=device)

        the_input = torch.cat([note, x], dim=1)

        # creates hidden layer values
        h0, c0, hconductor, cconductor, hdecoder, cdecoder = self.init_hidden(batch_size)

        # resets encoder at the beginning of every batch and gives it x
        x, hidden = self.encoder(x, (h0, c0))

        #print("x encoder", x.shape)
        x = self.encoderOut(x)

        mu_var_list = torch.chunk(x, 2 * m_key_count, dim=-1)

        # Split encoder outputs into a mean and variance vector
        mu_list, log_var_list = torch.chunk(x, 2, dim=-1)

        # Make sure that the log variance is positive
        log_var_list = softplus(log_var_list)

        with torch.no_grad():
            batch_size = mu_list.size(0)
            epsilon = torch.randn(batch_size, 1, self.latent_features * m_key_count)

            if use_cuda:
                epsilon = epsilon.cuda()

        # setting sigma
        sigma = torch.exp(log_var_list * 2)

        # generate z - latent space
        z = mu_list + epsilon * sigma

        #separate keys
        z_list = z.view(batch_size, note_length, m_key_count, -1)
        z_list = self.linear_z(z_list)

        #print("z_list", z_list.shape)

        #Vertial attention
        for i in range(note_length // NOTESPERBAR):
            z_horizontal = z_list[:,16 * i,:,:]
            for j in range(m_key_count):
                current_z = z_horizontal[:,j,:]
                other_z = torch.cat((z_horizontal[:,:j,:], z_horizontal[:,(j+1):,:]),1)

                #long short term memory for this key
                current_h_conducter = hconductor[:, :, j, :].contiguous()
                current_c_conducter = cconductor[:, :, j, :].contiguous()
                current_h_decoder = hdecoder[:, :, j, :].contiguous()
                current_c_decoder = cdecoder[:, :, j, :].contiguous()

                #Get simple attention
                attn_weights = self.attn_v(current_z, other_z)
                context = attn_weights.bmm(other_z).squeeze(1)
                conductor_input = torch.cat((current_z, context), dim = 1)

                #print("conductor_input", conductor_input.shape)
                # print("current_h", current_h.shape)
                # print("current_c", current_c.shape)


                embedding, (current_h_conducter, current_c_conducter) = self.conductor(conductor_input.unsqueeze(1), (current_h_conducter, current_c_conducter))
                embedding = embedding.expand(batch_size, NOTESPERBAR, embedding.shape[2])

                decoder_input = torch.cat([embedding, the_input[:, range(i * 16, i * 16 + 16), :]], dim=-1)
                #print("embedding", embedding.shape)
                #print("the_input[:, i, :]]", the_input.shape)

                notes_cur, (current_h_decoder, current_c_decoder) = self.decoder(decoder_input, (current_h_decoder, current_c_decoder))
                aux = self.linear(notes_cur)

                #print("notes_cur", notes_cur.shape)
                #print("aux", aux.shape)

                multi_notes[:,range(i * 16, i * 16 + 16),j,:] = aux
                notes[:, range(i * 16, i * 16 + 16), :] += aux #!!!!!


        #Horizontal attention
        #To do

        outputs = {}
        outputs["x_hat"] = notes
        outputs["z"] = z
        outputs["mu"] = mu_list
        outputs["log_var"] = log_var_list
        outputs["multi_notes"] = multi_notes

        return outputs



if __name__ == "__main__":
    vae_cell = VAECell(latent_features)
    if use_cuda:
        vae_cell = vae_cell.cuda()
    a = torch.randn(64, 256, 61, device = device)
    vae_cell(a)