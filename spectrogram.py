import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_spectrogram(file_path, bpm=None, bars=16, log_scale=True, use_mel=False):
    
    """
    Supply file path to an mp3
    Optional:
      - known BPM (or will try to estimate)
      - bars (increments of 16 on x axis as default)
      - log scale (default to True for clearer visualization)
      - mel spectrograpm applies a Fourier transformation on audio signals windowed in time, to transform from time domain to frequency domain 
      (default is short-time Fourier transform (STFT), used to analyze how the frequency content of a nonstationary signal changes over time. The magnitude squared of the STFT is known as the spectrogram time-frequency representation of the signal.)
    """
    y, sr = librosa.load(file_path, sr=None)  # 'sr=None' keeps the original sampling rate

    # if no bpm provided, estimate it first
    if bpm == None:
      bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
      print('Estimated tempo: {:.2f} beats per minute'.format(bpm))

    # compute the spectrogram (mel or STFT)
    if use_mel:
        # Mel spectrogram for smoother visualization across frequency scales
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=512, n_mels=128)
        D = librosa.power_to_db(S, ref=np.max)
    else:
        # STFT spectrogram
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2  # Increased n_fft for finer resolution
        D = librosa.power_to_db(S, ref=np.max)

    # calculate bars based on bpm create x-axis ticks for bars
    beats_per_bar = 4
    samples_per_beat = sr * 60 / bpm
    samples_per_bar = samples_per_beat * beats_per_bar
    time_per_bar = samples_per_bar / sr
    total_bars = int(np.floor(len(y) / samples_per_bar))
    labels = [f"{i*bars}" for i in range(0, total_bars // bars + 1)]
    ticks = [i * bars * time_per_bar for i in range(0, total_bars // bars + 1)]

    # plot spectrogram figure
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log' if log_scale else 'hz', hop_length=512)

    # adjusting the x-axis
    plt.xticks(ticks, labels)

    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{file_path} Spectrogram')
    plt.xlabel('Bars')
    plt.ylabel('Frequency (log scale)' if log_scale else 'Frequency')
    plt.show()
