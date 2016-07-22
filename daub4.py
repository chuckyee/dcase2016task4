#!/usr/bin/env python

from __future__ import print_function, division

import librosa
import numpy as np
import csv, os
from scipy import signal


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

def daubechies_transform(y, topscale = 5, return_chunks = False):
    assert(len(y) % 2**topscale == 0)
    sig = signal.daub(2)  # Daub4 transform
    wav = sig * (-1)**np.arange(len(sig))
    z = []
    for n in range(topscale):
        a1 = np.sum(sig[i]*np.roll(y, -i) for i in range(len(sig)))[::2]
        d1 = np.sum(wav[i]*np.roll(y, -i) for i in range(len(sig)))[::2]
        z.append(d1)
        y = a1
    z.append(y)
    z.reverse()
    return z if return_chunks else np.hstack(z)

def main():
    # output data files
    daub4_filename  = "dcase2016.daub4"
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

    # how many times to apply Daub4 wavelet transform
    # for 16kHz dataset, 5x gives length 2000 trends
    topscale = 6

    # Do we have scikits.samplerate library for fast resampling?
    print('HAS_SAMPLERATE: ', librosa.core.audio._HAS_SAMPLERATE)


    cd = ChimeDataset(root)

    # Get list of wav files for selected data set
    fileheads = cd.load_fileheads(dataset)

    daub4s = []
    labels = []
    for filehead in fileheads:
        y, sr = cd.load_audio(filehead, sampling_rate)
        print("Processing {}: {} samples at {}Hz".format(filehead, len(y), sr))
        z = daubechies_transform(y, topscale, return_chunks = True)
        daub4s.append(z[0])

        annotations = cd.load_annotations(filehead)
        # only keep labels which at least 2 out of 3 annotators agreed upon
        labels.append(annotations["majorityvote"])

    np.savetxt(daub4_filename, daub4s)

    with open(labels_filename, 'w') as f:
        f.write("\n".join(labels))
        
if __name__ == '__main__':
    main()
