import librosa
import numpy as np
import torch
from models import *
import soundfile
import librosa
import pyrubberband as rb

def wav2spectrum(fname):
    x, sr = librosa.load(fname)
    S = librosa.stft(x, num_fft) # num_fft = 512
    p = np.angle(S)

    S = np.log1p(np.abs(S))
    return S, sr

def fileToSpectrum(file, sr):
    S = librosa.stft(file, num_fft)
    p = np.angle(S)

    S = np.log1p(np.abs(S))
    return S, sr

def spectrum2wav(spectrum, sr, outfile):
    # Return the all-zero vector with the same shape of `a_content`
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, num_fft))
    #librosa.output.write_wav(outfile, x, sr) # Depr
    soundfile.write(outfile, x, sr)

def compute_content_loss(a_C, a_G):
    """
    Compute the content cost
    Arguments:
    a_C -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)
    Returns:
    J_content -- scalar that you compute using equation 1 above
    """
    m, n_C, n_H, n_W = a_G.shape

    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    a_C_unrolled = a_C.view(m * n_C, n_H * n_W)
    a_G_unrolled = a_G.view(m * n_C, n_H * n_W)

    # Compute the cost
    J_content = 1.0 / (4 * m * n_C * n_H * n_W) * torch.sum((a_C_unrolled - a_G_unrolled) ** 2)

    return J_content

def compute_content_loss_two(a_C, a_G):
    """
    Compute the content cost
    Arguments:
    a_C -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)
    Returns:
    J_content -- scalar that you compute using equation 1 above
    """
    m, n_C = a_G.shape

    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    a_C_unrolled = a_C.view(m * n_C)
    a_G_unrolled = a_G.view(m * n_C)

    # Compute the cost
    J_content = 1.0 / (4 * m * n_C) * torch.sum((a_C_unrolled - a_G_unrolled) ** 2)

    return J_content



def gram(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_L)
    Returns:
    GA -- Gram matrix of shape (n_C, n_C)
    """
    GA = torch.matmul(A, A.t())

    return GA


def gram_over_time_axis(A):
    """
    Argument:
    A -- matrix of shape (1, n_C, n_H, n_W)
    Returns:
    GA -- Gram matrix of A along time axis, of shape (n_C, n_C)
    """
    m, n_C, n_H, n_W = A.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    A_unrolled = A.view(m * n_C * n_H, n_W)
    GA = torch.matmul(A_unrolled, A_unrolled.t())

    return GA

def gram_over_time_axis_two(A):
    """
    Argument:
    A -- matrix of shape (1, n_C, n_H, n_W)
    Returns:
    GA -- Gram matrix of A along time axis, of shape (n_C, n_C)
    """
    m, n_C = A.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    A_unrolled = A.view(m * n_C)
    GA = torch.matmul(A_unrolled, A_unrolled.t())

    return GA


def compute_layer_style_loss(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)
    Returns:
    J_style_layer -- tensor representing a scalar style cost.
    """
    m, n_C, n_H, n_W = a_G.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)

    # Calculate the gram
    # !!!!!! IMPORTANT !!!!! Here we compute the Gram along n_C,
    # not along n_H * n_W. But is the result the same? No.
    GS = gram_over_time_axis(a_S)
    GG = gram_over_time_axis(a_G)

    # Computing the loss
    J_style_layer = 1.0 / (4 * (n_C ** 2) * (n_H * n_W)) * torch.sum((GS - GG) ** 2)

    return J_style_layer

def compute_layer_style_loss_two(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)
    Returns:
    J_style_layer -- tensor representing a scalar style cost.
    """
    m, n_C = a_G.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)

    # Calculate the gram
    # !!!!!! IMPORTANT !!!!! Here we compute the Gram along n_C,
    # not along n_H * n_W. But is the result the same? No.
    GS = gram_over_time_axis_two(a_S)
    GG = gram_over_time_axis_two(a_G)

    # Computing the loss
    J_style_layer = 1.0 / (4 * (n_C ** 2)) * torch.sum((GS - GG) ** 2)

    return J_style_layer


def changeOutputTempo(song, targetSong, sr):
    estimatedTempo = librosa.beat.tempo(song, sr)
    outputTempo = librosa.beat.tempo(targetSong, sr)
    ratio = estimatedTempo / outputTempo
    outsong = rb.time_stretch(song, sr, ratio)
    newtempo = librosa.beat.tempo(outsong, sr)
    return outsong