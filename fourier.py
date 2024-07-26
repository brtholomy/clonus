import visualization as viz

import click
import audioflux as af
import numpy as np
import matplotlib.pyplot as plt

from audioflux.type import (SpectralFilterBankScaleType, SpectralFilterBankStyleType,
                            WindowType, SpectralDataType)
from audioflux.utils import power_to_db
from audioflux.display import fill_spec


def GetBFT(path):
    audio_arr, sr = af.read(path)
    # num : num of frequency bins, meaning a FFT is performed for that f envelope
    # radix2_exp : fft_length=2**radix2_exp : default 2^12 = 4096
    # slide_length : data sampling size, by default = fft_length / 4
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

def CompareToPink(path):
    pink_obj, pink_audio_arr, pink_spec_dB_arr = ShortTimeFourierTransform(PINKPATH)
    obj, audio_arr, spec_dB_arr = ShortTimeFourierTransform(path)
    # change length so that we can simply diff the two
    spec_dB_arr_fitted = np.resize(spec_dB_arr, pink_spec_dB_arr.shape)
    diff = pink_spec_dB_arr - spec_dB_arr_fitted
    print(f'{diff.shape = }')
    # TODO: this isn't the right graph:
    plt.plot(diff)
    viz.PlotShow()

@click.command()
@click.option(
    '--file_path', '-f',
    default='./samps/pink.wav',
    type=str,
    show_default=True,
    help='wav file to analyze'
)
@click.option(
    '--compare_to_pink', '-c',
    is_flag=True,
    default=False,
    help='generate a diff plot against pink noise reference'
)
def main(file_path, compare_to_pink):
    obj, audio_arr, spec_dB_arr = ShortTimeFourierTransform(file_path)
    PlotSpectrogram(obj, audio_arr, spec_dB_arr)
    if compare_to_pink:
        CompareToPink(file_path)

if __name__ == "__main__":
    main()
