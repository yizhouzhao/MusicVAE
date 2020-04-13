from src.params import *
from src.model import VariationalAutoencoder
import matplotlib.pyplot as plt
from src.midi_builder import MidiBuilder

builder = MidiBuilder()

''' 
Create model and load state dict from path
'''
def loadModel(path):
    model = VariationalAutoencoder(latent_features, TEACHER_FORCING, learning_rate)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    return model

'''
Show reconstructions of the model
'''
def showReconstructionsMultiNotes(multi_notes : torch.Tensor, x : torch.Tensor):
    '''
    :param multi_notes:  reconstructed multinotes
    :param x: original notes
    :return:
    '''
    multi_notes_np = x_hat.detach().numpy()


    for i, seq in enumerate(multi_notes_np):
        midi_out = np.zeros((multi_notes_np.shape[0],multi_notes_np.shape[2]))
        for j in range(m_key_count):
            x_j_np = seq[:, j, :]
            row_maxes = seq.max(axis=1).reshape(-1, 1)
            midi_out_j = np.where(seq == row_maxes, 1, 0)
            midi_out += midi_out_j

            # if np.all(midi_out[:, -1]):
            #     print("Midi: {} is all silent at channel {}".format(i, j))
            #     continue

        midi = builder.midi_from_piano_roll(midi_out[:,:-1]) # Select all notes but the silent one
        plt.figure(figsize=(10, 3))
        plt.title("Midi {}".format(i))

        builder.plot_midi(midi)
        plt.savefig("midi/img_midi_{}.png".format(i))

        midi.write('midi/{}.mid'.format(i))

    # Compare to originals
    x_np = x.detach().numpy()

    for i, seq in enumerate(x_np):
        midi_out = seq

        if np.all(midi_out[:, -1]):
            print("Midi: {} is all silent".format(i))
            continue

        midi = builder.midi_from_piano_roll(midi_out[:, :-1])  # Select all notes but the silent one
        plt.figure(figsize=(10, 3))
        plt.title("Orig Midi {}".format(i))

        builder.plot_midi(midi)
        plt.savefig("midi/img_midi_{}_orig.png".format(i))

        midi.write('midi/{}_orig.mid'.format(i))


def showReconstructions(model, x_hat, x):
    x_hat_np = x_hat.detach().numpy()
    
    for i, seq in enumerate(x_hat_np):
        row_maxes = seq.max(axis=1).reshape(-1, 1)
        midi_out = np.where(seq == row_maxes, 1, 0)
        
        if np.all(midi_out[:,-1]):
            print("Midi: {} is all silent".format(i))
            continue
    
        np.savetxt("midi/csv_midi_out_{}.csv".format(i), midi_out, delimiter=";")

        midi = builder.midi_from_piano_roll(midi_out[:,:-1]) # Select all notes but the silent one
        plt.figure(figsize=(10, 3))
        plt.title("Midi {}".format(i))

        builder.plot_midi(midi)
        plt.savefig("midi/img_midi_{}.png".format(i))

        midi.write('midi/{}.mid'.format(i))

    # Compare to originals
    x_np = x.detach().numpy()

    for i, seq in enumerate(x_np):
        midi_out = seq

        if np.all(midi_out[:,-1]):
            print("Midi: {} is all silent".format(i))
            continue
        
        midi = builder.midi_from_piano_roll(midi_out[:,:-1]) # Select all notes but the silent one
        plt.figure(figsize=(10, 3))
        plt.title("Orig Midi {}".format(i))
        
        builder.plot_midi(midi)
        plt.savefig("midi/img_midi_{}_orig.png".format(i))

        midi.write('midi/{}_orig.mid'.format(i))


'''
Generate from the latent space
'''
def generateFromLatentSpace(model, gen_batch=10, showPlot=True):
    z_gen = torch.randn(gen_batch, 256, 32)
   
    # Sample from latent space
    h_gen,c_gen,hconductor_gen,cconductor_gen = model.init_hidden(gen_batch)
    conductor_hidden_gen = (hconductor_gen,cconductor_gen)

    notes_gen = torch.zeros(gen_batch,TOTAL_NOTES,NUM_PITCHES,device=device)

    # For the first timestep the note is the embedding
    note_gen = torch.zeros(gen_batch, 1 , NUM_PITCHES,device=device)

    counter=0

    for i in range(totalbars):
        decoder_hidden_gen = (torch.randn(1,gen_batch, decoders_initial_size,device=device), torch.randn(1,gen_batch, decoders_initial_size,device=device))
        embedding_gen, conductor_hidden_gen = model.conductor(z_gen[:,i,:].view(gen_batch,1, -1), conductor_hidden_gen)

        for _ in range(sequence_length):
            # Concat embedding with previous note

            e_gen = torch.cat([embedding_gen, note_gen], dim=-1)
            e_gen = e_gen.view(gen_batch, 1, -1)

            # Generate a single note (for each batch)
            note_gen, decoder_hidden_gen = model.decoder(e_gen, decoder_hidden_gen)

            aux_gen = model.linear(note_gen)

            aux_gen = torch.softmax(aux_gen, dim=2);
            #notes_gen[:,range(i*16,i*16+16),:]=aux_gen;
            notes_gen[:,counter,:] = aux_gen.squeeze();

            note_gen = aux_gen
            counter = counter+1

    if not showPlot:
        print(notes_gen)
    else:
        notes_np = notes_gen.cpu().detach().numpy()
 
        for i, seq in enumerate(notes_np):
            row_maxes = seq.max(axis=1).reshape(-1, 1)
            midi_out = np.where(seq == row_maxes, 1, 0)
            if np.all(midi_out[:,-1]):
                print("Midi: {} is all silent".format(i))
                continue

            np.savetxt("midi/gen_csv_midi_out_{}.csv".format(i), midi_out, delimiter=";")

            midi = builder.midi_from_piano_roll(midi_out[:,:-1]) # Select all notes but the silent one
            plt.figure(figsize=(10, 3))
            plt.title("Gen Midi {}".format(i))
            
            builder.plot_midi(midi)
            plt.savefig("midi/gen_img_midi_{}.png".format(i))

            midi.write('midi/gen_{}.mid'.format(i))


'''
Below is for testing
'''
if __name__ == "__main__":
    from data_utils import MidiDataset, BarTransform
    from torch.autograd import Variable #deprecated!!!

    transform = BarTransform(bars=totalbars, note_count=NUM_PITCHES)
    midi_dataset = MidiDataset(csv_file='piano_rolls_a_small_fraction.csv', transform = transform)
    dataset_size = len(midi_dataset)           #number of musics on dataset
    test_size = int(test_split * dataset_size) #test size length
    train_size = dataset_size - test_size      #train data length
    train_dataset, test_dataset = random_split(midi_dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=1)#, sampler=test_sampler)

    model = loadModel('records/net_Apr_9th.pt')

    x = next(iter(test_loader))
    x = Variable(x['piano_rolls'].type('torch.FloatTensor'))

    x = x.to(device)

    model.set_scheduled_sampling(1.)  # Please use teacher forcing for validations
    outputs = model(x)

    x_hat = outputs['x_hat']
    x = x.to("cpu")
    x_hat = x_hat.to("cpu")

    showReconstructions(model, x_hat, x)
    generateFromLatentSpace(model, 10, True)
