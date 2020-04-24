from src.data_utils import transposer, encode_dummies, chopster, trim_blanks, minister
import pretty_midi
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

print(os.getcwd())


def PreprocessMIDIPiano(midi_file: str, save_file: str, append_csv=False):
    '''
    Preprocess MIDI music file to data matrix for training
    :param
    midi_file:the name of the music file of MIDI format
    save_file: the name of the saving file
    :return:
    '''

    if not append_csv:
        saved_columns = [
            pretty_midi.note_number_to_name(n) for n in range(36, 96)
        ]
        piano_rolls = pd.DataFrame(columns=['piano_roll_name', 'timestep'] +
                                   saved_columns)
        piano_rolls = piano_rolls.set_index(['piano_roll_name', 'timestep'])
        piano_rolls.to_csv(save_file, sep=',', encoding='utf-8')

    #Identify the music key: Major or Minor
    #semi_shift = transposer(midi_file)
    #print("shift")
    #Read midi file
    pm = pretty_midi.PrettyMIDI(midi_file)

    #Get sample freqency for data matrix(piano roll) according to the Sixteenth note ??
    sampling_freq = 1 / (pm.get_beats()[1] / 4)

    # Only deal with the MIDI pieces that have one instrument as acoustic grand piano
    #print(len(pm.instruments) == 1)
    if len(pm.instruments) > 1:
        raise Exception(
            "Sorry, deal with the MIDI pieces that have one instrument")
    instrument = pm.instruments[0]
    #print(pretty_midi.program_to_instrument_name(int(instrument.program)))
    #assert int(instrument.program) == 0

    #if the music if a major, shift its scale to C major
    #if the music if a minor, shift its scale to A minor
    #for note in instrument.notes:
    #    note.pitch += semi_shift

    #encode midi to data frame
    df = encode_dummies(instrument, sampling_freq).fillna(value=0)

    #cut the data to record the music note only from c3 to b7
    df = chopster(df)

    #trim the beginning empty sound track
    df = trim_blanks(df)

    #Ignore the strength(velocity) and regard the sound as one and non-sound as zero
    df = minister(df)

    #organize data frame and save
    df.reset_index(inplace=True, drop=True)
    df['timestep'] = df.index
    df['piano_roll_name'] = midi_file
    df = df.set_index(['piano_roll_name', 'timestep'])
    #print(df.head())
    df.to_csv(save_file, sep=',', mode='a', encoding='utf-8', header=False)


def PreprocessMIDIPianoFiles(midi_files: list, save_file: str):
    '''
    Preprocess MIDI music files to csv file
    :param
    midi_files:the name of the music file of MIDI format
    save_file: the name of the saving file
    '''
    has_file = False
    for midi_file in tqdm(midi_files):
        #print(midi_file)
        if not has_file:
            PreprocessMIDIPiano(midi_file, save_file, False)
            has_file = True
        else:
            PreprocessMIDIPiano(midi_file, save_file, True)


if __name__ == "__main__":
    pass
    # midi_file = 'E:/researches/music_vae/data/2004\\MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
    #
    # PreprocessMIDIPiano(midi_file, "piano_roll.csv")
    #
    # rolls = pd.read_csv("piano_roll.csv", sep=',', index_col=['piano_roll_name', 'timestep'])
    # print(rolls.head())