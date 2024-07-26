import visualization as viz

import audioflux as af
import numpy as np
import matplotlib.pyplot as plt

from audioflux.type import (SpectralFilterBankScaleType, SpectralFilterBankStyleType,
                            WindowType, SpectralDataType)
from audioflux.utils import power_to_db
from audioflux.display import fill_spec


def GetBFT(path):
    audio_arr, sr = af.read(path)
    # Create BFT object of Linser(STFT)
    obj = af.BFT(num=2049, radix2_exp=12, samplate=sr, low_fre=0., high_fre=20_000.,
                 window_type=WindowType.HANN, slide_length=1024,
                 scale_type=SpectralFilterBankScaleType.LINEAR,
                 style_type=SpectralFilterBankStyleType.SLANEY,
                 data_type=SpectralDataType.POWER)
    return obj, audio_arr

def GetDbSpectrogram(obj, audio_arr):
    # Extract spectrogram of dB
    spec_arr = obj.bft(audio_arr)
    spec_arr = np.abs(spec_arr)
    spec_dB_arr = power_to_db(spec_arr)
    return spec_dB_arr

def PlotSpectrogram(obj, audio_arr, spec_dB_arr):
    # Show spectrogram plot
    audio_len = audio_arr.shape[-1]
    fig, ax = plt.subplots()
    img = fill_spec(spec_dB_arr, axes=ax,
                    x_coords=obj.x_coords(audio_len),
                    y_coords=obj.y_coords(),
                    x_axis='time', y_axis='log',
                    title='BFT-Linear Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    viz.PlotShow()

def ShortTimeFourierTransform(path):
    obj, audio_arr = GetBFT(path)
    spec_dB_arr = GetDbSpectrogram(obj, audio_arr)
    return obj, audio_arr, spec_dB_arr

def main():
    path = './samps/pink.wav'
    obj, audio_arr, spec_dB_arr = ShortTimeFourierTransform(path)
    PlotSpectrogram(obj, audio_arr, spec_dB_arr)

if __name__ == "__main__":
    main()
