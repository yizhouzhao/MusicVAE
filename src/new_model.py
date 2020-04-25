from src.params import *

from torch_struct import NonProjectiveDependencyCRF

import time


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
        attn_energies = torch.zeros(this_batch_size, max_len)  # B x S

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
        self.encoder = torch.nn.LSTM(batch_first=True,
                                     input_size=input_size,
                                     hidden_size=enc_hidden_size,
                                     num_layers=1,
                                     bidirectional=True)

        # encoded data goes onto connect linear layer. inputs must be*2 because LSTM is bidirectional
        # output must be 2*latentspace because it needs to be split into miu and sigma right after.
        self.encoderOut = MLP(enc_hidden_size * 2,
                              [400, latent_features * 2 * m_key_count],
                              dropout_rate=dropout_rate)

        # after being converted data goes through a fully connected layer
        self.linear_z = nn.Linear(in_features=latent_features,
                                  out_features=decoders_initial_size)

        #conduction to transform z
        #decoders_initial_size + context_size(decoders_initial_size)\
        if use_dependency_tree_horizontal:
            self.conductor = nn.LSTM(decoders_initial_size * 3,
                                     decoders_initial_size,
                                     num_layers=1,
                                     batch_first=True)
        else:
            self.conductor = nn.LSTM(decoders_initial_size * 2,
                                     decoders_initial_size,
                                     num_layers=1,
                                     batch_first=True)

        self.decoder = nn.LSTM(NUM_PITCHES + decoders_initial_size,
                               decoders_initial_size,
                               num_layers=1,
                               batch_first=True)

        # Linear note to note type (classes/pitches)
        self.linear = nn.Linear(decoders_initial_size, NUM_PITCHES)

        #torch structure attention
        if use_dependency_tree_vertical:
            self.W_1_v = nn.Parameter(torch.randn(m_key_energy_dim,
                                                  decoders_initial_size),
                                      requires_grad=True)
            self.W_2_v = nn.Parameter(torch.randn(m_key_energy_dim,
                                                  decoders_initial_size),
                                      requires_grad=True)
            self.b_v = nn.Parameter(torch.zeros(m_key_energy_dim),
                                    requires_grad=True)
            self.s_v = nn.Parameter(torch.randn(m_key_energy_dim),
                                    requires_grad=True)

            # if use_cuda:
            #     self.W_1_v = self.W_1_v.cuda()
            #     self.W_2_v = self.W_2_v.cuda()
            #     self.b_v = self.b_v.cuda()
            #     self.s_v = self.s_v.cuda()

        if use_dependency_tree_horizontal:
            self.W_1_h = nn.Parameter(torch.randn(m_key_energy_dim,
                                                  decoders_initial_size),
                                      requires_grad=True)
            self.W_2_h = nn.Parameter(torch.randn(m_key_energy_dim,
                                                  decoders_initial_size),
                                      requires_grad=True)
            self.b_h = nn.Parameter(torch.zeros(m_key_energy_dim),
                                    requires_grad=True)
            self.s_h = nn.Parameter(torch.randn(m_key_energy_dim),
                                    requires_grad=True)

    # used to modify the parameter for dependency for every batch
    def modify_weight_and_bias_vertical(self, batch_size):
        w_1_v = self.W_1_v.unsqueeze(0).expand(batch_size, m_key_energy_dim,
                                               decoders_initial_size)
        w_2_v = self.W_2_v.unsqueeze(0).expand(batch_size, m_key_energy_dim,
                                               decoders_initial_size)
        b_v = self.b_v.unsqueeze(1).unsqueeze(0).expand(
            batch_size, m_key_energy_dim, m_key_count)
        s_v = self.s_v.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, m_key_energy_dim)

        return w_1_v, w_2_v, b_v, s_v

    def modify_weight_and_bias_horizontal(self, batch_size):
        w_1_h = self.W_1_h.unsqueeze(0).expand(batch_size, m_key_energy_dim,
                                               decoders_initial_size)
        w_2_h = self.W_2_h.unsqueeze(0).expand(batch_size, m_key_energy_dim,
                                               decoders_initial_size)
        b_h = self.b_h.unsqueeze(1).unsqueeze(0).expand(
            batch_size, m_key_energy_dim, m_key_count * totalbars)
        s_h = self.s_h.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, m_key_energy_dim)

        return w_1_h, w_2_h, b_h, s_h

    # used to initialize the hidden layer of the encoder to zero before every batch
    def init_hidden(self, batch_size):
        # must be 2 x batch x hidden_size because its a bi-directional LSTM
        init = torch.zeros(2, batch_size, enc_hidden_size, device=device)
        c0 = torch.zeros(2, batch_size, enc_hidden_size, device=device)

        # 2 because has 2 layers
        # n_layers_conductor
        init_conductor = torch.zeros(1,
                                     batch_size,
                                     m_key_count,
                                     decoders_initial_size,
                                     device=device)
        c_condunctor = torch.zeros(1,
                                   batch_size,
                                   m_key_count,
                                   decoders_initial_size,
                                   device=device)

        h_decoder = torch.zeros(1,
                                batch_size,
                                m_key_count,
                                decoders_initial_size,
                                device=device)
        c_decoder = torch.zeros(1,
                                batch_size,
                                m_key_count,
                                decoders_initial_size,
                                device=device)

        return init, c0, init_conductor, c_condunctor, h_decoder, c_decoder

    def forward(self, x):
        #add empty sound in the beginning
        batch_size = x.size(0)
        note_length = x.size(1)
        note = torch.zeros(batch_size, 1, NUM_PITCHES, device=device)

        #music notes record
        notes = torch.zeros(batch_size,
                            note_length,
                            NUM_PITCHES,
                            device=device)

        #music notes with multi_channel
        multi_notes = torch.zeros(batch_size,
                                  note_length,
                                  m_key_count,
                                  NUM_PITCHES,
                                  device=device)

        the_input = torch.cat([note, x], dim=1)

        # creates hidden layer values
        h0, c0, hconductor, cconductor, hdecoder, cdecoder = self.init_hidden(
            batch_size)

        # create weights for structure attention
        if use_dependency_tree_vertical:
            w_1_v, w_2_v, b_v, s_v = self.modify_weight_and_bias_vertical(
                batch_size)

        if use_dependency_tree_horizontal:
            w_1_h, w_2_h, b_h, s_h = self.modify_weight_and_bias_horizontal(
                batch_size)

        # resets encoder at the beginning of every batch and gives it x
        x, hidden = self.encoder(x, (h0, c0))

        #print("x encoder", x.shape)
        x = self.encoderOut(x)

        # This is never used
        # mu_var_list = torch.chunk(x, 2 * m_key_count, dim=-1)

        # Split encoder outputs into a mean and variance vector
        mu_list, log_var_list = torch.chunk(x, 2, dim=-1)

        # Make sure that the log variance is positive
        log_var_list = softplus(log_var_list)

        with torch.no_grad():
            batch_size = mu_list.size(0)
            epsilon = torch.zeros(batch_size, 1,
                                  self.latent_features * m_key_count)

            if use_cuda:
                epsilon = epsilon.cuda()

        # setting sigma
        sigma = torch.exp(log_var_list * 2)

        # generate z - latent space
        z = mu_list + epsilon * sigma

        #separate keys
        z_list = z.view(batch_size, note_length, m_key_count, -1)
        z_list = self.linear_z(z_list)

        #print("z_list", z_list.shape, NOTESPERBAR, note_length // NOTESPERBAR)

        if use_dependency_tree_horizontal:
            z_horizontal_bars = z_list[:,
                                       range(0, note_length, NOTESPERBAR
                                             ), :, :]
            '''
            Replace for loop to get log potentials
            '''
            z_horizontal_bars_reshaped = z_horizontal_bars.view(
                batch_size, -1, decoders_initial_size)
            z_horizontal_bars_reshaped = z_horizontal_bars_reshaped.transpose(
                1, 2)
            w1_times_hi = torch.bmm(
                w_1_h,
                z_horizontal_bars_reshaped)  #batch x energy_dim x (bars * keys)
            w1_times_hi = w1_times_hi + b_h
            w2_times_hj = torch.bmm(
                w_2_h,
                z_horizontal_bars_reshaped)  # batch x energy_dim x (bars * keys)

            #reshape
            w1_times_hi = w1_times_hi.view(batch_size, m_key_energy_dim,
                                           m_key_count, totalbars)
            w2_times_hj = w2_times_hj.view(batch_size, m_key_energy_dim,
                                           m_key_count, totalbars)

            #print("w1_times_hi", w1_times_hi.shape)

            #expand
            w1_times_hi = w1_times_hi.unsqueeze(-1).expand(
                batch_size, m_key_energy_dim, m_key_count, totalbars,
                totalbars)
            w2_times_hj = w2_times_hj.unsqueeze(-2).expand(
                batch_size, m_key_energy_dim, m_key_count, totalbars,
                totalbars)

            #plus
            w1_plus_w2 = torch.tanh(w1_times_hi + w2_times_hj)

            w1_plus_w2 = w1_plus_w2.view(batch_size, m_key_energy_dim, -1)

            #log potential
            log_potentials_h = torch.tanh(torch.bmm(
                s_h, w1_plus_w2)).squeeze(1).view(batch_size, m_key_count,
                                                  totalbars, totalbars)

            #print("log_potentials_h", log_potentials_h.shape)

            # log_potentials_h = torch.zeros(batch_size, m_key_count, totalbars, totalbars, device = device)
            #
            # for j in range(m_key_count):
            #     print(j)
            #     z_horizontal_bars_j = z_horizontal_bars[:,:,j,:]
            #     for b in range(batch_size):
            #         for i_h in range(totalbars):
            #             h_i = z_horizontal_bars_j[b, i_h, :]
            #             for j_h in range(totalbars):
            #                 h_j = z_horizontal_bars_j[b, j_h, :]
            #                 # print("h_i",h_i.shape)
            #                 # print("h_j", h_j.shape)
            #                 # print("self.W_1_h", self.W_1_h.shape)
            #                 # print("self.W_2_h", self.W_2_h.shape)
            #                 # print("self.b_h", self.b_h.shape)
            #                 temp = torch.tanh(self.W_1_h.matmul(h_i) + self.W_2_h.matmul(h_j) + self.b_h)
            #                 log_potentials_h[b, j, i_h, j_h] = torch.tanh(
            #                                     torch.dot(self.s_h,
            #                                               temp
            #                                               )
            #                                 )

            # print("z_horizontal_bars_j", z_horizontal_bars_j.shape)
            # print("log_potentials_h", log_potentials_h[0,0])
            # exit()

        #Vertial attention
        for i in range(note_length // NOTESPERBAR):
            z_horizontal = z_list[:, NOTESPERBAR * i, :, :]

            if use_dependency_tree_vertical:
                '''
                Substitude for loop for get lop potentials
                '''
                #begin_time = int(round(time.time() * 1000))

                z_horizontal_t = z_horizontal.transpose(1, 2)

                w1_times_hi = torch.bmm(w_1_v, z_horizontal_t)
                w1_times_hi = w1_times_hi + b_v
                w2_times_hj = torch.bmm(w_2_v, z_horizontal_t)
                w1_times_hi = w1_times_hi.unsqueeze(-1).expand(
                    batch_size, m_key_energy_dim, m_key_count, m_key_count)
                w2_times_hj = w2_times_hj.unsqueeze(-2).expand(
                    batch_size, m_key_energy_dim, m_key_count, m_key_count)

                w1_plus_w2 = torch.tanh(w1_times_hi + w2_times_hj)
                w1_plus_w2 = w1_plus_w2.view(batch_size, m_key_energy_dim,
                                             m_key_count * m_key_count)
                log_potentials_v = torch.tanh(
                    torch.bmm(s_v, w1_plus_w2).squeeze(1).view(
                        batch_size, m_key_count, m_key_count))
                # end_time  = int(round(time.time() * 1000))
                #
                # print("begin time", begin_time)
                # print("end time: ", end_time)
                # print(log_potentials)

                # log_potentials = torch.zeros(batch_size, m_key_count,
                #                             m_key_count, device = device)
                #
                # for b in range(batch_size):
                #     for i_d in range(m_key_count):
                #         h_i = z_horizontal[b, i_d, :]
                #         for j_d in range(m_key_count):
                #             h_j = z_horizontal[b, j_d, :]
                #             # print("h_i",h_i.shape)
                #             # print("h_j", h_j.shape)
                #             # print("self.W_1_v", self.W_1_v.shape)
                #             # print("self.W_2_v", self.W_2_v.shape)
                #             # print("self.b_v", self.b_v.shape)
                #             temp = torch.tanh(self.W_1_v.matmul(h_i) + self.W_2_v.matmul(h_j) + self.b_v)
                #             log_potentials[b, i_d, j_d] = torch.tanh(
                #                 torch.dot(self.s_v,
                #                           temp
                #                           )
                #             )
                #
                # end_time = int(round(time.time() * 1000))
                # print("spend time: ", end_time - begin_time)
                # print(log_potentials)
                # exit()

                dist_v = NonProjectiveDependencyCRF(log_potentials_v)

                # if np.random.rand() < 0.005:
                #     print("z_horizontal", z_horizontal[0])
                #     print("w1, w2 ", self.W_1_v[0, :10], self.W_2_v[0, :10])
                #     print("b_v, s_v", self.b_v, self.s_v)
                #     print("log potentials vertiacal", log_potentials_v[0])
                #     print("dist",dist_v.marginals[0])

                for j in range(m_key_count):
                    current_z = z_horizontal[:, j, :]
                    current_h_conducter = hconductor[:, :, j, :].contiguous()
                    current_c_conducter = cconductor[:, :, j, :].contiguous()
                    current_h_decoder = hdecoder[:, :, j, :].contiguous()
                    current_c_decoder = cdecoder[:, :, j, :].contiguous()
                    context_v = dist_v.marginals[:, :, j].unsqueeze(1).bmm(
                        z_horizontal).squeeze(1)

                    if use_dependency_tree_horizontal:
                        z_horizontal_bars_j = z_horizontal_bars[:, :, j, :]
                        dist_h = NonProjectiveDependencyCRF(
                            log_potentials_h[:, j, :, :].contiguous())
                        context_h = dist_h.marginals[:, :, i].unsqueeze(1).bmm(
                            z_horizontal_bars_j).squeeze(1)
                        #print("z_horizontal_bars_j", z_horizontal_bars_j.shape)
                        # if np.random.rand() < 0.05:
                        #     print("log potentials horizontal", log_potentials_h[0,j])
                        conductor_input = torch.cat(
                            (current_z, context_h, context_v), dim=1)

                    else:
                        conductor_input = torch.cat((current_z, context_v),
                                                    dim=1)

                    #print("context_v", context_v.shape)
                    #print("conductor_input", conductor_input.shape)

                    embedding, (current_h_conducter,
                                current_c_conducter) = self.conductor(
                                    conductor_input.unsqueeze(1),
                                    (current_h_conducter, current_c_conducter))

                    embedding = embedding.expand(batch_size, NOTESPERBAR,
                                                 embedding.shape[2])

                    decoder_input = torch.cat([
                        embedding, the_input[:,
                                             range(i * 16, i * 16 + 16), :]
                    ],
                                              dim=-1)
                    #print("embedding", embedding.shape)
                    #print("the_input[:, i, :]]", the_input.shape)
                    #exit()

                    notes_cur, (current_h_decoder,
                                current_c_decoder) = self.decoder(
                                    decoder_input,
                                    (current_h_decoder, current_c_decoder))
                    aux = self.linear(notes_cur)
                    aux = torch.softmax(aux, dim=2)

                    # print("notes_cur", notes_cur.shape)
                    # print("aux", aux.shape)

                    multi_notes[:, range(i * 16, i * 16 + 16), j, :] = aux
                    #notes[:, range(i * 16, i * 16 + 16), :] += aux / m_key_count  # !!!!!
            else:  # use normal attention
                for j in range(m_key_count):
                    current_z = z_horizontal[:, j, :]
                    # long short term memory for this key
                    current_h_conducter = hconductor[:, :, j, :].contiguous()
                    current_c_conducter = cconductor[:, :, j, :].contiguous()
                    current_h_decoder = hdecoder[:, :, j, :].contiguous()
                    current_c_decoder = cdecoder[:, :, j, :].contiguous()
                    if use_attention:
                        other_z = torch.cat((z_horizontal[:, :j, :],
                                             z_horizontal[:, (j + 1):, :]), 1)

                        #Get simple attention
                        attn_weights = self.attn_v(current_z, other_z)
                        context = attn_weights.bmm(other_z).squeeze(1)
                        conductor_input = torch.cat((current_z, context),
                                                    dim=1)

                        #print("conductor_input", conductor_input.shape)
                        # print("current_h", current_h.shape)
                        # print("current_c", current_c.shape)
                        embedding, (current_h_conducter,
                                    current_c_conducter) = self.conductor(
                                        conductor_input.unsqueeze(1),
                                        (current_h_conducter,
                                         current_c_conducter))

                    else:
                        context = torch.randn(
                            (batch_size, decoders_initial_size), device=device)
                        conductor_input = torch.cat((current_z, context),
                                                    dim=1)

                    embedding, (current_h_conducter,
                                current_c_conducter) = self.conductor(
                                    conductor_input.unsqueeze(1),
                                    (current_h_conducter, current_c_conducter))

                    embedding = embedding.expand(batch_size, NOTESPERBAR,
                                                 embedding.shape[2])

                    decoder_input = torch.cat([
                        embedding, the_input[:,
                                             range(i * 16, i * 16 + 16), :]
                    ],
                                              dim=-1)
                    #print("embedding", embedding.shape)
                    #print("the_input[:, i, :]]", the_input.shape)

                    notes_cur, (current_h_decoder,
                                current_c_decoder) = self.decoder(
                                    decoder_input,
                                    (current_h_decoder, current_c_decoder))
                    aux = self.linear(notes_cur)
                    aux = torch.softmax(aux, dim=2)

                    #print("notes_cur", notes_cur.shape)
                    #print("aux", aux.shape)

                    multi_notes[:, range(i * 16, i * 16 + 16), j, :] = aux
                    notes[:, range(i * 16, i * 16 +
                                   16), :] += aux / m_key_count  #!!!!!

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
    a = torch.randn(64, 256, 61, device=device)
    vae_cell(a)