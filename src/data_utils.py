import pandas as pd
import numpy as np
import pretty_midi
import music21

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

    dframe.drop(labels=[pretty_midi.note_number_to_name(n) for n in range(108, 128)], axis=1, inplace=True)
    dframe.drop(labels=[pretty_midi.note_number_to_name(n) for n in range(0, 48)], axis=1, inplace=True)
    return dframe


# Non-zero values changed to 1's
def minister(dframe):
    return dframe.where(dframe<1, 1)


# DISCLAIMER:
# This file is inspired by Nick Kelly by his article on tranposing MIDI files.
# http://nickkellyresearch.com/python-script-transpose-midi-files-c-minor/
# Transpose MIDI to same key (C major or A minor)

def transposer(midi_file):
    midi_array = midi_file.split('/')

    # converting everything into the key of C major or A minor
    # Major conversion
    majors = dict([("A-", 4),("G#", 4),("A", 3),("A#", 2),("B-", 2),("B", 1),("C", 0),("C#", -1),("D-", -1),("D", -2),("D#", -3),("E-", -3),("E", -4),("F", -5),("F#", 6),("G-", 6),("G", 5)])
    # Minor conversion
    minors = dict([("G#", 1), ("A-", 1),("A", 0),("A#", -1),("B-", -1),("B", -2),("C", -3),("C#", -4),("D-", -4),("D", -5),("D#", 6),("E-", 6),("E", 5),("F", 4),("F#", 3),("G-", 3),("G", 2)])

    score = music21.converter.parse(midi_file)
    key = score.analyze('key')
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]

    else: # key.mode == "minor":
        halfSteps = minors[key.tonic.name]

    return halfSteps


