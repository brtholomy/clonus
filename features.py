import visualization as viz

import numpy as np
import audioflux as af
from audioflux.type import SpectralDataType, SpectralFilterBankScaleType

import matplotlib.pyplot as plt
from audioflux.display import fill_wave, fill_plot, fill_spec

# Read audio data and sample rate
samp_path = './samps/pink.wav'
audio_arr, sr = af.read(samp_path)
audio_len = audio_arr.shape[-1]

bft_obj = af.BFT(num=256, samplate=sr, radix2_exp=12, slide_length=1024,
                 data_type=SpectralDataType.MAG,
                 scale_type=SpectralFilterBankScaleType.LINEAR)
spec_arr = bft_obj.bft(audio_arr)
spec_arr = np.abs(spec_arr)

# Create Spectral object and extract spectral feature
spectral_obj = af.Spectral(num=bft_obj.num,
                           fre_band_arr=bft_obj.get_fre_band_arr())
spectral_obj.set_time_length(spec_arr.shape[-1])

flatness_arr = spectral_obj.flatness(spec_arr)
novelty_arr = spectral_obj.novelty(spec_arr)
entropy_arr = spectral_obj.entropy(spec_arr)
rms_arr = spectral_obj.rms(spec_arr)
slope_arr = spectral_obj.slope(spec_arr)

# Display
fig, ax = plt.subplots(nrows=7, figsize=(8, 10), sharex=True)
times = np.arange(0, flatness_arr.shape[-1]) * (bft_obj.slide_length / bft_obj.samplate)

fill_wave(audio_arr, samplate=sr, axes=ax[0])
fill_spec(spec_arr, axes=ax[1],
          x_coords=bft_obj.x_coords(audio_len), y_coords=bft_obj.y_coords(),
          y_axis='log')
fill_plot(times, flatness_arr, axes=ax[2], label='flatness')
fill_plot(times, novelty_arr, axes=ax[3], label='novelty')
fill_plot(times, entropy_arr, axes=ax[4], label='entropy')
fill_plot(times, rms_arr, axes=ax[5], label='rms_arr')
fill_plot(times, slope_arr, axes=ax[6], label='slope_arr')

viz.PlotShow()
