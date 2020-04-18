from src.params import *
from src.model import VariationalAutoencoder
from src.data_utils import MidiDataset, BarTransform
from src.loss import  ELBO_loss, ELBO_loss2, ELBO_loss_Multi
from src.new_model import VAECell

import time
import os
import math

from torch.autograd import Variable #deprecated!!!

if __name__ == "__main__":
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

    torch.save(net.state_dict(),'records/net_Apr_18_2_chord.pt')