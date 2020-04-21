from src.params import *
from src.model import VariationalAutoencoder
from src.data_utils import MidiDataset, BarTransform
from src.loss import  ELBO_loss, ELBO_loss2, ELBO_loss_Multi
from src.new_model import VAECell

from torch.utils.tensorboard import SummaryWriter

import time
from datetime import datetime

import os
import math

from torch.autograd import Variable #deprecated!!!

if __name__ == "__main__":
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("date and time:", date_time)

    writer = SummaryWriter("runs/" + date_time)

    params_dict = {"NOTESPERBAR":NOTESPERBAR, "totalbars":totalbars, "NUM_PITCHES" : NUM_PITCHES,
                   "batch_size":batch_size, "learning_rate":learning_rate, "num_epochs":num_epochs,
                   "m_key_count":m_key_count, "use_new_model":use_new_model, "use_attention":use_attention,
                   "use_dependency_tree_vertical":use_dependency_tree_vertical, "use_dependency_tree_horizontal":use_dependency_tree_horizontal,
                   "use_permutation_loss":use_permutation_loss
                   }

    writer.add_hparams(params_dict, {})

    #cut the music piece into several bars, each bar comtains equal number of note sequences
    transform = BarTransform(bars=totalbars, note_count=NUM_PITCHES)#configures number of input bars

    #Load dataset
    midi_dataset = MidiDataset(csv_file=data_file, transform = transform) #imports dataset
    print("Train.py Memory Usage for Training Data: ", midi_dataset.get_mem_usage(),"MB")

    #Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    dataset_size = len(midi_dataset)           #number of musics on dataset
    test_size = int(test_split * dataset_size) #test size length
    train_size = dataset_size - test_size      #train data length

    #Split dataset into training/testing
    train_dataset, test_dataset = random_split(midi_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=1)#, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=1)#, sampler=test_sampler)

    print("Train.py Train size: {}, Test size: {}".format(train_size, test_size))

    #Model
    if use_new_model:
        net = VAECell(latent_features)
    else:
        net = VariationalAutoencoder(latent_features, TEACHER_FORCING, eps_i = 1)

    if use_cuda:
        net = net.cuda()

    print("Train.py The model looks like this:\n", net)

    # define our optimizer
    # The Adam optimizer works really well with VAEs.
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if use_new_model:
        loss_function = ELBO_loss_Multi
    else:
        loss_function = ELBO_loss2

    #Learning rate: warmup and decay
    warmup_lerp = 1/warmup_epochs

    if warmup_epochs > num_epochs - pre_warmup_epochs:
        warmup_epochs=num_epochs - pre_warmup_epochs

    warmup_w=0

    scheduled_decay_rate = 40

    def lin_decay(i, mineps=0):
        return np.max([mineps, 1 - (1/len(train_loader))*i])

    def inv_sigmoid_decay(i, rate=40):
        return rate/(rate + np.exp(i/rate))

    eps_i = 1
    use_scheduled_sampling = False

    train_loss, valid_loss = [], []
    train_kl, valid_kl,train_klw = [], [],[]

    start = time.time()

    # epochs loop
    for epoch in range(num_epochs):
        print("Training epoch {}".format(epoch))
        batch_loss, batch_kl, batch_klw = [], [], []
        net.train()

        running_loss = 0.0
        kl_loss = 0.0
        tpr = 0.0
        tnr = 0.0
        ppv = 0.0
        npv = 0.0
        for i_batch, sample_batched in enumerate(train_loader):
            if i_batch % 5 == 0:
                print("i_batch", i_batch)
            #    break
            x = sample_batched['piano_rolls']

            x = x.type('torch.FloatTensor')

            # if i_batch%10==0:
            #    print("batch:",i_batch)

            x = Variable(x)

            # This is an alternative way of putting
            # a tensor on the GPU
            x = x.to(device)

            ## Calc the sched sampling rate:
            if epoch >= pre_warmup_epochs and use_scheduled_sampling:
                eps_i = inv_sigmoid_decay(i_batch, rate=scheduled_decay_rate)

            if use_new_model:
                pass
            else:
                net.set_scheduled_sampling(eps_i)

            outputs = net(x)
            mu, log_var = outputs['mu'], outputs['log_var']

            #elbo, kl, kl_w = loss_function(x_hat, x, mu, log_var, warmup_w, with_logits=False)
            if use_new_model:
                multi_notes = outputs["multi_notes"]
                elbo, kl, kl_w = loss_function(multi_notes, x, mu, log_var, warmup_w)
            else:
                x_hat = outputs['x_hat']
                elbo, kl, kl_w = loss_function(x_hat, x, mu, log_var, warmup_w)

            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

            batch_loss.append(elbo.item())
            batch_kl.append(kl.item())
            batch_klw.append(kl_w.item())

            #loss
            running_loss += elbo.item()
            kl_loss += kl.item()
            if i_batch % log_frequency == log_frequency - 1:
                # ...log the running loss
                writer.add_scalar('ELBO loss',
                                  running_loss / log_frequency,
                                  epoch * len(train_loader) + i_batch)
                writer.add_scalar('KL loss',
                                  kl_loss / log_frequency,
                                  epoch * len(train_loader) + i_batch)
                running_loss = 0.0
                kl_loss = 0.0

                writer.add_scalar('TPR',
                                  tpr / log_frequency,
                                  epoch * len(train_loader) + i_batch)
                tpr = 0.0
                writer.add_scalar('TNR',
                                  tnr / log_frequency,
                                  epoch * len(train_loader) + i_batch)
                tnr = 0.0
                writer.add_scalar('PPV',
                                  ppv / log_frequency,
                                  epoch * len(train_loader) + i_batch)
                ppv = 0.0
                writer.add_scalar('NPV',
                                  npv / log_frequency,
                                  epoch * len(train_loader) + i_batch)
                npv = 0.0

            x_real = x.view(-1, NUM_PITCHES)
            x_index = torch.arange(x_real.size(0), requires_grad=False)
            x_hat = torch.zeros_like(x_real)
            if use_new_model:
                multi_notes = outputs["multi_notes"]
                multi_notes = multi_notes.view(-1, m_key_count, NUM_PITCHES) + 1e-6

                for j in range(m_key_count):
                    multi_notes_j = multi_notes[:, j, :]
                    max_index = torch.argmax(multi_notes_j, dim=-1)
                    x_hat[x_index, max_index] = 1

            else:
                notes = outputs['x_hat']
                notes = notes.view(-1, NUM_PITCHES) + 1e-6

                max_index = torch.argmax(notes, dim=-1)
                x_hat[x_index, max_index] = 1

            tpr += torch.mean(x_hat[x_real == 1]).item()  # true positive rate
            tnr += (1 - torch.mean(x_hat[x_real == 0]).item())  # true negative rate
            ppv += torch.mean(x_real[x_hat == 1]).item()  # postive predictive rate
            npv += (1 - torch.mean(x_real[x_hat == 0]).item())  # negative predictive rate

            # print("what is wrong with: ")
            # print(torch.mean(x_hat[x_real == 1]).item())
            # print(torch.mean(x_hat[x_real == 0]).item())
            # print(torch.mean(x_real[x_hat == 1]).item())
            # print(torch.mean(x_real[x_hat == 0]).item())

        train_loss.append(np.mean(batch_loss))
        train_kl.append(np.mean(batch_kl))
        train_klw.append(np.mean(batch_klw))

        # Evaluate, do not propagate gradients
        with torch.no_grad():
            net.eval()

            # Just load a single batch from the test loader
            x = next(iter(test_loader))
            x = Variable(x['piano_rolls'].type('torch.FloatTensor'))

            x = x.to(device)

            if use_new_model:
                pass
            else:
                net.set_scheduled_sampling(1)

            outputs = net(x)
            x_hat = outputs['x_hat']
            mu, log_var = outputs['mu'], outputs['log_var']
            z = outputs["z"]


            if use_new_model:
                multi_notes = outputs["multi_notes"]
                elbo, kl, klw = loss_function(multi_notes, x, mu, log_var, warmup_w)
            else:
                elbo, kl, klw = loss_function(x_hat, x, mu, log_var, warmup_w)

            # We save the latent variable and reconstruction for later use
            # we will need them on the CPU to plot
            #x = x.to("cpu")
            #x_hat = x_hat.to("cpu")
            #z = z.detach().to("cpu").numpy()

            valid_loss.append(elbo.item())
            valid_kl.append(kl.item())

        if epoch >= pre_warmup_epochs:
            warmup_w = warmup_w + warmup_lerp
            if warmup_w > 1:
                warmup_w = 1.

        #if epoch == 0:
        #    continue

        print("train_loss:", train_loss[-1], np.mean(train_loss))
        print("valid_loss:", valid_loss[-1], np.mean(valid_loss))

    torch.save(net.state_dict(),"records/" + date_time + ".pt")
    writer.close()
