import librosa
import numpy as np
from models import *
import soundfile
import librosa
import pyrubberband as rb
from utils import changeOutputTempo, wav2spectrum, spectrum2wav

nightcall, sr1 = librosa.load("input_nightcall.wav")
stairway, sr2 = librosa.load("input_stairway.wav")

def changeOutputTempo(song, targetSong, sr):
    estimatedTempo = librosa.beat.tempo(song, sr)
    outputTempo = librosa.beat.tempo(targetSong, sr)
    print(estimatedTempo, outputTempo)
    ratio = estimatedTempo / outputTempo
    #song = rb.change_tempo(song, sr, estimatedTempo, final_tempo)
    outsong = rb.time_stretch(song, sr, ratio)
    newtempo = librosa.beat.tempo(outsong, sr)
    return outsong

song = changeOutputTempo(nightcall, stairway, sr1)
print(nightcall.shape, song.shape, stairway.shape)

#soundfile.write(song, sr1, "song.wav")
soundfile.write("song.wav", song, sr1)