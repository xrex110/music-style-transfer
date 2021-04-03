import librosa
import numpy as np
import torch
from models import *
import soundfile
import librosa
import pyrubberband as rb
from utils import changeOutputTempo, wav2spectrum, spectrum2wav

nightcall, sr1 = wav2spectrum("input_nightcall.wav")
stairway, sr2 = wav2spectrum("input_stairway.wav")

song = changeOutputTempo(nightcall, stairway, sr1)

spectrum2wav(song, sr1, "song.wav")