#!/usr/bin/env python

from __future__ import print_function, division

import librosa
import numpy as np
import csv, os


class ChimeDataset:
    def __init__(self, root = "."):
        self.root = root
    
    def load_fileheads(self, dataset):
        INDEX, FILENAME = range(2)
        index_file = os.path.join(self.root, "chime_home", "{}.csv".format(dataset))
        with open(index_file, 'r') as f:
            csv_reader = csv.reader(f)
            fileheads = [line[FILENAME] for line in csv_reader]
        return fileheads
    
    def load_audio(self, filehead, sampling_rate):
        # load audio file without resampling
        full_filename = "{}.{}.wav".format(filehead, sampling_rate)
        audio_path = os.path.join(self.root, 'chime_home', 'chunks', full_filename)
        return librosa.load(audio_path, sr = None)
    
    def load_annotations(self, filehead):
        csv_filename = "{}.csv".format(filehead)
        csv_path = os.path.join(self.root, 'chime_home', 'chunks', csv_filename)
        annotations = {}
        with open(csv_path, 'r') as f:
            csv_reader = csv.reader(f)
            for k,v in csv_reader:
                annotations[k] = v
        return annotations

def main():
    # output data files
    mfcc_filename   = "dcase2016.mfcc"
    labels_filename = "dcase2016.labels"

    # directory where CHiME dataset has been extracted
    root = "/Users/chuckyee/Documents"

    # dataset is one of:
    #   development_chunks_raw
    #   development_chunks_refined
    #   development_chunks_refined_crossval_dcase2016
    #   evaluation_chunks_refined
    dataset = "development_chunks_refined"

    # sampling_rate is either "48kHz" or "16kHz"
    # the DCASE2016 challenge is based on the 16kHz downsampled mono files
    sampling_rate = '16kHz'

    # number of samples in short-time FFT window
    # good values: 4096 for 48kHz, 1024 for 16kHz
    n_fft = 1024

    # number of samples to shift FFT window
    # good values: 2048 for 48kHz, 512 for 16kHz
    hop_length = 512

    # number of MFCC coefficients to return
    # choose power of 3 because we are using 3-leg isometries in our MERA
    n_mfcc = 27

    # Do we have scikits.samplerate library for fast resampling?
    print('HAS_SAMPLERATE: ', librosa.core.audio._HAS_SAMPLERATE)


    cd = ChimeDataset(root)

    # Get list of wav files for selected data set
    fileheads = cd.load_fileheads(dataset)

    mfccs  = []
    labels = []
    for filehead in fileheads:
        y, sr = cd.load_audio(filehead, sampling_rate)
        print("Processing {}: {} samples at {}Hz".format(filehead, len(y), sr))
        mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = n_mfcc,
                                    n_fft = n_fft, hop_length = hop_length)
        mfccs.append(mfcc.T.flatten())

        annotations = cd.load_annotations(filehead)
        # only keep labels which at least 2 out of 3 annotators agreed upon
        labels.append(annotations["majorityvote"])

    np.savetxt(mfcc_filename, mfccs)

    with open(labels_filename, 'w') as f:
        f.write("\n".join(labels))
        
if __name__ == '__main__':
    main()
