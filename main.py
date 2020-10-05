from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
from load_data import (
    Voc, unicodeToAscii, normalizeString, readVocs, filterPair,
    filterPairs, loadPrepareData, PAD_token, SOS_token,EOS_token,
    MAX_LENGTH, MIN_COUNT, trimRareWords
)
def main():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    save_dir = os.path.join("data", "save")
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    # Trim voc and pairs
    pairs = trimRareWords(voc, pairs, MIN_COUNT)

# run main thread
if __name__ == "__main__":
    main()
