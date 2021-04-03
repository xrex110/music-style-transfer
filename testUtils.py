import librosa
import numpy as np
import torch
from models import *
import soundfile
import librosa
import pyrubberband as rb
from utils import changeOutputTempo, wav2spectrum, spectrum2wav

nightcall, sr1 = librosa.load("input_nightcall.wav")
stairway, sr2 = librosa.load("input_stairway.wav")

song = changeOutputTempo(nightcall, stairway, sr1)

soundfile.write("song.wav", song, sr1)