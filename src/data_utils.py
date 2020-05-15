import pandas as pd
import numpy as np
import pretty_midi
import music21
from src.params import m_midi_start, m_midi_end


def encode_dummies(instrument, sampling_freq):
    """ Gonna cheat a little bit by transposing the instrument piano roll.
        However, that leaves us with a lot of blank space.
        Coercing to type uint8, since the max velocity is 127, and there are also only 128 notes. uint8 goes to 255.
        Saves memory
    """
    note_columns = [pretty_midi.note_number_to_name(n) for n in range(0, 128)]
    pr = instrument.get_piano_roll(fs=sampling_freq).astype('uint8').T
    return pd.DataFrame(pr, columns=note_columns)


def trim_blanks(df):
    """
        Remove the first period of no activity (fast forward to where the first note begins for this instrument)
    """
    nonzero = df.apply(lambda s: s != 0)
    nonzeroes = df[nonzero].apply(pd.Series.first_valid_index)
    first_nonzero = nonzeroes.min()
    if first_nonzero is np.nan:
        return None
    return df.iloc[int(first_nonzero):]


# Chops off the upperbound and lowerbound of zeros
# The lower bound note is set a C, which might make it easier
# to make the MIDI play in the same key.
def chopster(dframe):
    # Figure out range of frame (0-128)
    df_max = dframe.max(axis=0)

    dframe.drop(
        labels=[pretty_midi.note_number_to_name(n) for n in range(m_midi_end, 128)],
        axis=1,
        inplace=True)
    dframe.drop(
        labels=[pretty_midi.note_number_to_name(n) for n in range(0, m_midi_start)],
        axis=1,
        inplace=True)
    return dframe


# Non-zero values changed to 1's
def minister(dframe):
    return dframe.where(dframe < 1, 1)


# DISCLAIMER:
# This file is inspired by Nick Kelly by his article on tranposing MIDI files.
# http://nickkellyresearch.com/python-script-transpose-midi-files-c-minor/
# Transpose MIDI to same key (C major or A minor)


def transposer(midi_file):
    midi_array = midi_file.split('/')

    # converting everything into the key of C major or A minor
    # Major conversion
    majors = dict([("A-", 4), ("G#", 4), ("A", 3), ("A#", 2), ("B-", 2),
                   ("B", 1), ("C", 0), ("C#", -1), ("D-", -1), ("D", -2),
                   ("D#", -3), ("E-", -3), ("E", -4), ("F", -5), ("F#", 6),
                   ("G-", 6), ("G", 5)])
    # Minor conversion
    minors = dict([("G#", 1), ("A-", 1), ("A", 0), ("A#", -1), ("B-", -1),
                   ("B", -2), ("C", -3), ("C#", -4), ("D-", -4), ("D", -5),
                   ("D#", 6), ("E-", 6), ("E", 5), ("F", 4), ("F#", 3),
                   ("G-", 3), ("G", 2)])

    score = music21.converter.parse(midi_file)
    key = score.analyze('key')
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]

    else:  # key.mode == "minor":
        halfSteps = minors[key.tonic.name]

    return halfSteps


from torch.utils.data import Dataset, DataLoader


class MidiDataset(Dataset):
    """Pre-processed MIDI dataset."""
    def __init__(self, csv_file, transform, midi_start=48, midi_end=108):
        """
        Args:
            csv_file (string): Path to the csv file with piano rolls per song.
            transform (callable): Transform to be applied on a sample, is expected to implement "get_sections".
            midi_start (int): The first midi note in the dataset
            midi_end (int): The last midi note in the dataset
        """

        dtypes = {'piano_roll_name': 'str', 'timestep': 'uint32'}
        column_names = [
            pretty_midi.note_number_to_name(n)
            for n in range(midi_start, midi_end)
        ]
        for column in column_names:
            dtypes[column] = 'uint8'

        self.piano_rolls = pd.read_csv(
            csv_file,
            sep=',',
            index_col=['piano_roll_name', 'timestep'],
            dtype=dtypes)
        self.transform = transform

        self.init_dataset()

    def init_dataset(self):
        """
            Sets up an array containing a pd index (the song name) and the song section,
            ie. [("Song Name:1", 0), ("Song Name:1", 1), ("Song Name:1", 2)]
            for use in indexing a specific section
        """
        indexer = self._get_indexer()

        self.index_mapper = []
        for i in indexer:
            split_count = self.transform.get_sections(
                len(self.piano_rolls.loc[i].values))
            for j in range(0, split_count):
                self.index_mapper.append((i, j))

    def __len__(self):
        return len(self.index_mapper)

    def get_mem_usage(self):
        """
            Returns the memory usage in MB
        """
        return self.piano_rolls.memory_usage(deep=True).sum() / 1024**2

    def _get_indexer(self):
        """
            Get an indexer that treats each first level index as a sample.
        """
        return self.piano_rolls.index.get_level_values(0).unique()

    def __getitem__(self, idx):
        """
            Our frame is multi-index, so we're thinking each song is a single sample,
            and getting the individual bars is a transform of that sample?
        """
        song_name, section = self.index_mapper[idx]

        # Add a column for silences
        piano_rolls = self.piano_rolls.loc[song_name].values
        silence_col = np.zeros((piano_rolls.shape[0], 1))
        piano_rolls_with_silences = np.append(piano_rolls, silence_col, axis=1)

        # Transform the sample (including padding)
        sample = piano_rolls_with_silences.astype('float')
        sample = self.transform(sample)[section]

        # Fill in 1's for the silent rows
        empty_rows = ~sample.any(axis=1)
        if len(sample[empty_rows]) > 0:
            sample[empty_rows, -1] = 1.

        sample = {'piano_rolls': sample}

        return sample


import math


class BarTransform():
    def __init__(self, bars=1, note_count=60):
        self.split_size = bars * 16
        self.note_count = note_count

    def get_sections(self, sample_length):
        return math.ceil(sample_length / self.split_size)

    def __call__(self, sample):
        sample_length = len(sample)

        # Pad the sample with 0's if there's not enough to create equal splits into n bars
        leftover = sample_length % self.split_size
        if leftover != 0:
            padding_size = self.split_size - leftover
            padding = np.zeros((padding_size, self.note_count))
            sample = np.append(sample, padding, axis=0)

        sections = self.get_sections(sample_length)
        # Split into X equal sections
        split_list = np.array_split(sample, indices_or_sections=sections)

        return split_list