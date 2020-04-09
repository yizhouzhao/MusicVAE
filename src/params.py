import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import softplus
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split

'''
FILE INFO
'''
data_file = 'E:/researches/MusicVAE/training_data/piano_rolls_1.csv' #training dataset file

'''
DEVICE INFO
'''
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu") #device

'''
TRAINING SETTINGS
'''
random_seed = 42
test_split = .2
shuffle = True

'''
MODEL SETTINGS
'''
NOTESPERBAR=16 #total notes in one bar
totalbars=16 #total bars as input
NUM_PITCHES=60+1 # all possible notes to play +1 for silences
TOTAL_NOTES=NOTESPERBAR*totalbars
num_features=NUM_PITCHES #size of input feature vector
batch_size = 64 #actual batchsize
TEACHER_FORCING=True #not used but it will be needed
# define size variables
input_size = NUM_PITCHES
enc_hidden_size=256 #hidden size of encoder
conductor_hidden_size=256 #hidden size of decoder
decoders_hidden_size=64 #hidden size of decoder
decoders_initial_size=32 #decoder input size
n_layers_conductor=2 #not being used rn cuz number of layers is incorrect
n_layers_decoder=3 #not being used rn cuz number of layers is incorrect
latent_features=64 #latent space dimension
sequence_length = 16 #notes per decoder
dropout_rate = 0.2

'''
TRAIN
'''
learning_rate = 1e-3
num_epochs = 5
warmup_epochs = 90
pre_warmup_epochs = 10