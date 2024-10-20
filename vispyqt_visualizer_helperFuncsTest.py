import time
import sys
import math
import scipy
import numpy as np
from pylsl import local_clock
from mne.filter import create_filter
from scipy.signal import lfilter, lfilter_zi
from mne.time_frequency import tfr_array_morlet


# print full numpy
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

# https://www.codegrepper.com/code-examples/python/find+the+closest+value+in+an+array+python
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# === Create Line Variables
def create_line_variables(window, n_chan):
    line_data_output = []
    line_data_buffer = np.swapaxes(np.asarray([np.arange(window), np.zeros(window)]), 0, 1)
    for channel in range(0, n_chan):
        line_data_output.append(line_data_buffer)
    line_data_output = np.asarray(line_data_output)

    return line_data_output

# === Create Filter Variables
# filter_list = [[3, 40], [1, 8], [6, 15], [15, 30], [25, 50]]
def create_data_filter_variables(window, n_chan, sampling_rate, filter_list):
    af_buffer = [1.0]
    bf_output, zi_output, filt_output = [], [], []
    data_output, lsl_output, data_filter_output = np.zeros((window, n_chan)), np.zeros(window), np.zeros((len(filter_list), window, n_chan))
    lsl_output_filt_corrected, lsl_output_filt_index_corrected = np.zeros((len(filter_list), window)), np.zeros(len(filter_list))
    for filt in range(0, len(filter_list)):
        bf_buffer = create_filter(data_output.T, sampling_rate, filter_list[filt][0], filter_list[filt][1], method='fir', fir_design='firwin')
        zi_buffer = lfilter_zi(bf_buffer, af_buffer)
        filt_state_buffer = np.tile(zi_buffer, (n_chan, 1)).transpose()
        bf_output.append(bf_buffer)
        zi_output.append(zi_buffer)
        filt_output.append(filt_state_buffer.tolist())
        lsl_output_filt_index_corrected[filt] = int(len(bf_output[filt]) / 2)

    return data_filter_output, lsl_output_filt_corrected, lsl_output_filt_index_corrected, bf_output, zi_output, filt_output

# === TF Separate Bands: Multi/Single Plot with Separate/Average Channel
# input ex: window = window_eeg, n_chan = n_chan_eeg, band_range = np.array([[1, 3], [3, 8], [8, 13], [13, 30], [25, 50]]), filter_list = [[3, 40], [1, 8], [6, 15], [15, 30], [25, 50]]
def create_tf_line_variables(window, n_chan, filter_list, band_range):
    line_data_tf_buffer = np.swapaxes(np.asarray([np.arange(window), np.zeros(window)]), 0, 1)
    # === TF Multi Plot: Separate Band, Separate Channel
    line_data_eeg_tf_multiplot_output = []
    for channel in range(0, n_chan):
        for band in range(0, len(band_range)):
            line_data_eeg_tf_multiplot_output.append(line_data_tf_buffer)
    line_data_eeg_tf_multiplot_output = np.asarray(line_data_eeg_tf_multiplot_output)
    # === TF Single Plot: Separate Band, Average Channel
    line_data_eeg_tf_singleplot_output = []
    for band in range(0, len(band_range)):
        line_data_eeg_tf_singleplot_output.append(line_data_tf_buffer)
    line_data_eeg_tf_singleplot_output = np.asarray(line_data_eeg_tf_singleplot_output)

    return line_data_eeg_tf_multiplot_output, line_data_eeg_tf_singleplot_output










def get_artifact_mask_from_time_2023(artifact_time, artifact_window):
    artifact_mask_output = np.zeros([len(artifact_time), artifact_window])
    for channel in range(0, len(artifact_mask_output)):
        if len(artifact_time[channel]) > 0:
            for artifact in range(0, len(artifact_time[channel]), 2):
                if artifact_time[channel][artifact + 1] > (len(artifact_mask_output[channel]) - 1):
                    artifact_mask_output[channel][artifact_time[channel][artifact]:] = 1
                else:
                    artifact_mask_output[channel][artifact_time[channel][artifact]:(artifact_time[channel][artifact + 1] + 1)] = 1

    return artifact_mask_output

def get_artifact_time_from_mask_2023(artifact_mask):
    artifact_time_output = [[], [], [], []]
    for channel in range(0, len(artifact_mask)):
        if sum(artifact_mask[channel]) > 0:
            for sample in range(0, len(artifact_mask[channel])):
                if artifact_mask[channel][sample] == 1:
                    if sample == 0 or sample == len(artifact_mask[channel]) - 1:
                        artifact_time_output[channel].append(sample)
                    else:
                        if artifact_mask[channel][sample - 1] == 0 or artifact_mask[channel][sample + 1] == 0:
                            artifact_time_output[channel].append(sample)

    return artifact_time_output














# === 2023 artifact_time_eeg_index
def get_artifact_time_index_raw(artifact_time, artifact_data_range):
    # === raw no correction
    artifact_time_index = []
    for channel in range(0, len(artifact_time)):
        if len(artifact_time[channel]) > 0:
            artifact_index_buffer = (np.asarray(artifact_time[channel]) + artifact_data_range[0])
            artifact_time_index.append(artifact_index_buffer.tolist())
        else:
            artifact_time_index.append([])

    return artifact_time_index

# === 2023 artifact_time_eeg_index
# bf_index_offset[filt] = int(len(self.bf_eeg_all[filt]) / 2) where len(bf_eeg_index_offset) = n_filt + 1 where raw is bf_eeg_index_offset[0] = 0
def get_artifact_time_index_corrected(artifact_time, artifact_data_range, bf_index_offset):
    # === filter correction
    artifact_time_index_corrected = []
    for filt in range(1, len(bf_index_offset)):
        artifact_time_index_buffer = []
        for channel in range(0, len(artifact_time)):
            if len(artifact_time[channel]) > 0:
                artifact_buffer = (np.asarray(artifact_time[channel]) + int(bf_index_offset[filt]) + artifact_data_range[0])
                artifact_time_index_buffer.append(artifact_buffer.tolist())
            else:
                artifact_time_index_buffer.append([])
        artifact_time_index_corrected.append(artifact_time_index_buffer)

    return artifact_time_index_corrected


# len(full_artifact_mask) = window + np.max(bf_index_offset)
# update full_artifact_mask with new artifacts, NOTE: we must have ALREADY NP.ROLL full_artifact_mask, this function simply updates with new artifacts
def update_full_artifact_mask(full_artifact_mask, artifact_time_index, bf_index_offset):
    for channel in range(0, len(artifact_time_index)):
        if len(artifact_time_index[channel]) > 0:
            for this_artifact in range(0, len(artifact_time_index[channel]), 2):
                artifact_start_buffer = int(artifact_time_index[channel][this_artifact] + np.max(bf_index_offset))
                artifact_end_buffer = int(artifact_time_index[channel][this_artifact + 1] + np.max(bf_index_offset))
                full_artifact_mask[channel][artifact_start_buffer:artifact_end_buffer] = 1

    return full_artifact_mask





# === 2023 artifact_time_eeg_index
def get_artifact_time_index_raw_bandpower(artifact_time, artifact_data_range):
    artifact_time_index = []
    for channel in range(0, len(artifact_time)):
        if len(artifact_time[channel]) > 0:
            artifact_index_buffer = (np.asarray(artifact_time[channel]) + artifact_data_range[0])
            artifact_time_index.append(artifact_index_buffer.tolist())
        else:
            artifact_time_index.append([])

    return artifact_time_index








# input: data[channel][freq][time], band_freq[lower, uppder], returns subset of input data array with lower/upper freq limites to extract bands
# output: data[channel][freq][time]
def get_time_frequency_singleband(power, freqs, band_freq):
    freq_index = [np.where(freqs == find_nearest(freqs, band_freq[0]))[0][0], np.where(freqs == find_nearest(freqs, band_freq[1]))[0][0]]
    freqs_trim = freqs[freq_index[0]:freq_index[1] + 1]
    bandpower_trim = power[:, freq_index[0]:freq_index[1] + 1]

    return bandpower_trim, freqs_trim

# input: data[channel][freq][time]
# output: data[channel][time]
def get_time_frequency_overtime(spect_power, spect_freq, band_freq):
    spect_power_buffer, spect_freq_buffer = get_time_frequency_singleband(spect_power, spect_freq, band_freq) # trim spect_power to min/max of band_range
    spect_power_singleband_overtime = np.zeros([len(spect_power), len(spect_power[0][0])])
    for channel in range(0, len(spect_power)):
        for time in range(0, len(spect_power[0][0])):
            spect_power_singleband_overtime[channel][time] = np.mean(np.swapaxes(spect_power_buffer, 1, 2)[channel][time])

    return spect_power_singleband_overtime

# input: data[channel][freq][time]
# output: data[channel][band][time]
def get_time_frequency_bandpower_overtime(tf_power, tf_freq, band_range):
    tf_bandpower_overtime = np.zeros([len(tf_power), len(band_range), len(tf_power[0][0])])
    for band in range(0, len(band_range)):
        tf_power_buffer, spect_freq_buffer = get_time_frequency_singleband(tf_power, tf_freq, band_range[band])
        for channel in range(0, len(tf_power)):
            for time in range(0, len(tf_power[0][0])):
                tf_bandpower_overtime[channel][band][time] = np.mean(np.swapaxes(tf_power_buffer, 1, 2)[channel][time])

    return tf_bandpower_overtime





# === TF SCIPY SPECTRAL POWER NEW

# === TF SCIPY SPECTRAL POWER
# input: raw_data_eeg[channel][time], raw_lsl_eeg[time]
# output: tf_bandpower_trim_raw_overtime[time], tf_time_lsl is converted to lsl times
def get_bandpower_scipy_spectrogram_2023(raw_data_eeg, raw_lsl_eeg, band_range, sampling_rate, nfft, nperseg):
    # === SPECT: scipy.signal.spectrogram (scaling='density' default, or 'spectrum'), input[channel][time], output[channel][freq][time]
    total_freq = [np.min(band_range), np.max(band_range)]
    tf_freq, tf_time, tf_power = scipy.signal.spectrogram(raw_data_eeg, fs=sampling_rate, nperseg=nperseg, nfft=nfft)   # scaling='density' is default, or 'spectrum'
######    tf_power = 10*np.log10(tf_power)                                                                                  # === optional scale to DB using 10*np.log10(data), sometimes produces nan, may need to remove them
####    tf_power = 10*np.log10(abs(tf_power))                                                                               # === optional scale to DB using 10*np.log10(data), sometimes produces nan, may need to remove them
##    tf_power = 10*np.log10(tf_power)                                                                               # === optional scale to DB using 10*np.log10(data), sometimes produces nan, may need to remove them
    tf_power_trim, tf_freq_trim = get_time_frequency_singleband(tf_power, tf_freq, total_freq)                          # === trim tf_power, tf_freq to frequencies of interest to limit computing power
    # convert time to lsl for tf_power_trim
    tf_time_samples = tf_time * sampling_rate
    tf_time_lsl = np.zeros(len(tf_time_samples))                                                                        # convert spect_time to samples (INDEX) and LSL time
    for value in range(0, len(tf_time_samples)):
        tf_time_lsl[value] = raw_lsl_eeg[int(tf_time_samples[value])]
    # === TF RAW: bandpower OVERTIME from tf_power_trim
    tf_bandpower_trim_raw_overtime = np.zeros([len(band_range), len(raw_data_eeg), len(tf_time)])
    for band in range(0, len(band_range)):
        tf_bandpower_trim_raw_overtime_buffer = get_time_frequency_overtime(tf_power_trim, tf_freq_trim, band_range[band])
        for channel in range(0, len(raw_data_eeg)):
            tf_bandpower_trim_raw_overtime[band][channel] = tf_bandpower_trim_raw_overtime_buffer[channel]

    return tf_bandpower_trim_raw_overtime, tf_time_lsl, tf_time_samples

# === TF SCIPY SPECTRAL POWER
# input: raw_data_eeg[channel][time], raw_lsl_eeg[time]
def get_bandpower_scipy_spectrogram_interp_2023_NEW(raw_data_eeg, raw_lsl_eeg, band_range, sampling_rate, nfft, nperseg, interp_type, artifactTime, output_type):
    # output_type = "raw" (artifact reject only, still returns interp mask) or "interp" (interpolate: returns mask and interpolates tf data)
    # === SPECT: scipy.signal.spectrogram (scaling='density' default, or 'spectrum'), input[channel][time], output[channel][freq][time]
    total_freq = [np.min(band_range), np.max(band_range)]
    tf_freq, tf_time, tf_power = scipy.signal.spectrogram(raw_data_eeg, fs=sampling_rate, nperseg=nperseg, nfft=nfft)   # scaling='density' is default, or 'spectrum'
##    tf_power = 10*np.log10(tf_power)                                                                                  # === optional scale to DB using 10*np.log10(data), sometimes produces nan, may need to remove them
    tf_power_trim, tf_freq_trim = get_time_frequency_singleband(tf_power, tf_freq, total_freq)                          # === trim tf_power, tf_freq to frequencies of interest to limit computing power
    # convert time to lsl for tf_power_trim
    tf_time_samples = tf_time * sampling_rate
    tf_time_lsl = np.zeros(len(tf_time_samples))                                                                        # convert spect_time to samples (INDEX) and LSL time
    for value in range(0, len(tf_time_samples)):
        tf_time_lsl[value] = raw_lsl_eeg[int(tf_time_samples[value])]

    # ====== Interpolation basic variables
    # get tf_artifact_mask for interp vars, interpolate ACROSS CHANNEL tf_power_trim[channel][frequency][time] using tf_artifact_mask[channel][time]
    tf_artifact_mask = get_artifact_mask(artifactTime, tf_time_samples)
    tf_artifact_mask = connect_artifact_mask(tf_artifact_mask, 1)                # 1 = connect artifact times +/- 1
    tf_artifact_mask = reject_channel_artifact_mask(tf_artifact_mask, 0.5)       # 0.5 = 50% percent good data
    tf_power_trim_interp = interpolate_tf_power_singleband_across_channel(tf_power_trim, tf_artifact_mask, str(interp_type))

    # ====== Bandpower Interpolation tree
    tf_bandpower_interp_overtime_clean = np.zeros([len(band_range), len(raw_data_eeg), len(tf_time)])
    # === TF INTERP: bandpower OVERTIME from tf_power_trim
    for band in range(0, len(band_range)):
        # calculate tf_power_trim_interp overtime averaged across band range frequencies
        tf_power_trim_interp_overtime = get_time_frequency_overtime(tf_power_trim_interp, tf_freq_trim, band_range[band])
        # DOUBLE CHECK THAT THIS IS UPDATING BUT USING THE OLDER TF_ARTIFACT_MASK SO WE HAVE DIFFERENT MASKS FOR EACH BAND                
        # trim tf_power_trim_overtime_interp for outliers for the averaged band range by creating a new tf_artifact_mask, can replace with "mask_clean" to test == tf_artifact_mask
        tf_power_trim_overtime_interp_clean, tf_artifact_mask_clean = reject_tf_power_outliers_interpolated_overtime(tf_power_trim_interp_overtime, tf_artifact_mask, 2.5)
        # append data to the final tf_bandpower_interp_overtime_clean array
        for channel in range(0, len(raw_data_eeg)):
            tf_bandpower_interp_overtime_clean[band][channel] = tf_power_trim_overtime_interp_clean[channel]

    # === TF RAW: bandpower OVERTIME from tf_power_trim
    tf_bandpower_trim_raw_overtime = np.zeros([len(band_range), len(raw_data_eeg), len(tf_time)])
    for band in range(0, len(band_range)):
        tf_bandpower_trim_raw_overtime_buffer = get_time_frequency_overtime(tf_power_trim, tf_freq_trim, band_range[band])
        for channel in range(0, len(raw_data_eeg)):
            tf_bandpower_trim_raw_overtime[band][channel] = tf_bandpower_trim_raw_overtime_buffer[channel]

    if output_type == "raw":
        return tf_bandpower_trim_raw_overtime, tf_time_lsl, tf_time_samples, tf_artifact_mask_clean
    if output_type == "interp":
        return tf_bandpower_interp_overtime_clean, tf_time_lsl, tf_time_samples, tf_artifact_mask_clean

####    return tf_bandpower_trim_raw_overtime, tf_time_lsl, tf_time_samples, tf_artifact_mask_clean
##    return tf_bandpower_interp_overtime_clean, tf_time_lsl, tf_time_samples, tf_artifact_mask_clean




















# note: artifact_time and sample_time must be same format (e.g., sample # or lsl) but sample_time can be spectral or time-series
# input: artifact_time[channel][a1, a2, b1, b2, ...]
def get_artifact_mask(artifact_time, sample_time):
    artifact_mask = np.zeros([len(artifact_time), len(sample_time)])
    for channel in range(0, len(artifact_time)):
        if artifact_time[channel] != []:
            for this_artifact_time in range(0, len(artifact_time[channel]), 2):
                for this_sample_time in range(0, len(sample_time)):
                    if sample_time[this_sample_time] >= artifact_time[channel][this_artifact_time]:
                        if sample_time[this_sample_time] <= artifact_time[channel][this_artifact_time + 1]:
                            artifact_mask[channel][this_sample_time] = 1

    return artifact_mask

# connect artifacts separated by less than +/- connect_length in samples
# input: artifact_mask[channel][time], extend_length = +/- extend_length samples around artifacts
# output: same as input with +/- 1 time buffer
def connect_artifact_mask(artifact_mask, connect_length):
    # connect_length = 1            # default for time frequency
    # connect_length = 10            # default for time series by sample
    # forward check, ignore first connect_length samples
    for channel in range(0, len(artifact_mask)):
        for time in range(0, len(artifact_mask[channel]) - connect_length):
            if artifact_mask[channel][time] == 1:
                if artifact_mask[channel][time + 1] == 0:
                    if np.sum(artifact_mask[channel][time + 1:time + connect_length + 1]) > 0:
                        for artifact_time in range(time + 1, time + connect_length + 1):
                            if artifact_mask[channel][artifact_time] == 1:
                                artifact_mask[channel][time + 1:artifact_time] = 1
    # check first and last connect_length samples for edge cases
    for channel in range(0, len(artifact_mask)):
        for time in range(1, connect_length):
            if artifact_mask[channel][time] == 1:
                artifact_mask[channel][0:time] = 1
        for time in range(len(artifact_mask[channel]) - connect_length, len(artifact_mask[channel])):
            if artifact_mask[channel][time] == 1:
                artifact_mask[channel][time:len(artifact_mask[channel])] = 1

    return artifact_mask

# percent_good_data is how much of the data for each channel needs to be bad to mark entire channel as bad, default = .5 = 50%
def reject_channel_artifact_mask(artifact_mask, percent_good_data):
##    percent_good_data = 0.5               # default = 50% of data
    for channel in range(0, len(artifact_mask)):
        if (len(artifact_mask[channel]) - np.sum(artifact_mask[channel])) < int(len(artifact_mask[channel]) * percent_good_data):
            artifact_mask[channel] = 1

    return artifact_mask





# === TF SCIPY SPECTRAL POWER
# input: raw_data_eeg[channel][time], raw_lsl_eeg[time]
def get_bandpower_scipy_spectrogram_interp_2023(raw_data_eeg, raw_lsl_eeg, band_range, sampling_rate, nfft, nperseg, interp_type, artifactTime):
    # === SPECT: scipy.signal.spectrogram (scaling='density' default, or 'spectrum'), input[channel][time], output[channel][freq][time]
    total_freq = [np.min(band_range), np.max(band_range)]
    tf_freq, tf_time, tf_power = scipy.signal.spectrogram(raw_data_eeg, fs=sampling_rate, nperseg=nperseg, nfft=nfft)   # scaling='density' is default, or 'spectrum'
##    tf_power = 10*np.log10(tf_power)                                                                                  # === optional scale to DB using 10*np.log10(data), sometimes produces nan, may need to remove them
    tf_power_trim, tf_freq_trim = get_time_frequency_singleband(tf_power, tf_freq, total_freq)                          # === trim tf_power, tf_freq to frequencies of interest to limit computing power
    # convert time to lsl for tf_power_trim
    tf_time_samples = tf_time * sampling_rate
    tf_time_lsl = np.zeros(len(tf_time_samples))                                                                        # convert spect_time to samples (INDEX) and LSL time
    for value in range(0, len(tf_time_samples)):
        tf_time_lsl[value] = raw_lsl_eeg[int(tf_time_samples[value])]

    # ====== Interpolation basic variables
    # get tf_artifact_mask for interp vars, interpolate ACROSS CHANNEL tf_power_trim[channel][frequency][time] using tf_artifact_mask[channel][time]
    tf_artifact_mask = get_artifact_mask(artifactTime, tf_time_samples)
    tf_artifact_mask = connect_artifact_mask(tf_artifact_mask, 1)                # 1 = connect artifact times +/- 1
    tf_artifact_mask = reject_channel_artifact_mask(tf_artifact_mask, 0.5)       # 0.5 = 50% percent good data
    tf_power_trim_interp = interpolate_tf_power_singleband_across_channel(tf_power_trim, tf_artifact_mask, str(interp_type))

    # ====== Bandpower Interpolation tree
    tf_bandpower_interp_overtime_clean = np.zeros([len(band_range), len(raw_data_eeg), len(tf_time)])
    # === TF INTERP: bandpower OVERTIME from tf_power_trim
    for band in range(0, len(band_range)):
        # calculate tf_power_trim_interp overtime averaged across band range frequencies
        tf_power_trim_interp_overtime = get_time_frequency_overtime(tf_power_trim_interp, tf_freq_trim, band_range[band])
        # DOUBLE CHECK THAT THIS IS UPDATING BUT USING THE OLDER TF_ARTIFACT_MASK SO WE HAVE DIFFERENT MASKS FOR EACH BAND                
        # trim tf_power_trim_overtime_interp for outliers for the averaged band range by creating a new tf_artifact_mask, can replace with "mask_clean" to test == tf_artifact_mask
        tf_power_trim_overtime_interp_clean, tf_artifact_mask_clean = reject_tf_power_outliers_interpolated_overtime(tf_power_trim_interp_overtime, tf_artifact_mask, 2.5)
        # append data to the final tf_bandpower_interp_overtime_clean array
        for channel in range(0, len(raw_data_eeg)):
            tf_bandpower_interp_overtime_clean[band][channel] = tf_power_trim_overtime_interp_clean[channel]

    return tf_bandpower_interp_overtime_clean, tf_time_lsl, tf_time_samples, tf_artifact_mask_clean






# RECODE 7/20/22 JG, interpolate_tf_power_singleband_across_channel INTERPOLATES SEPARATELY AT EACH FREQUENCY
# description: corrects artifact flagged time points with median of good time points across channels separately for each FREQUENCY
# note: should work for singleband frequencies or bandpower averaged frequencies: just needs to be "overtime" with time component
# input: tf_artifact_mask[channel][artifact] where bad data artifact = 1 and good data = 0, interp_type: 'median', 'mean', mask (note mask sets = 1 for figure masking)
# output: tf_power with interpolated data
def interpolate_tf_power_singleband_across_channel(tf_power, tf_artifact_mask, interp_type):
    interp_label = [None]*4
    artifact_sum = int(np.sum(tf_artifact_mask))
    artifact_length = int(len(tf_artifact_mask) * len(tf_artifact_mask[0]))
    # check if all data are good or bad, if so then cannot interpolate
    if (artifact_sum != 0) and (artifact_sum != artifact_length):
        buffer_append = 0
        spect_power_median_buffer = np.zeros([len(tf_power[0]), artifact_length - artifact_sum])
        # get the median power of all good data for each frequency separately for interpolation
        for channel in range(0, len(tf_artifact_mask)):
            for time in range(0, len(tf_artifact_mask[channel])):
                if tf_artifact_mask[channel][time] == 0:
                    for frequency in range(0, len(tf_power[channel])):
                        spect_power_median_buffer[frequency][buffer_append] = tf_power[channel][frequency][time]
                    buffer_append += 1
        # calculate the median at each frequency to account for log power?
        spect_power_median = np.zeros(len(tf_power[0]))
        for frequency in range(0, len(spect_power_median)):
##            spect_power_median[frequency] = np.median(spect_power_median_buffer[frequency])
            if interp_type == 'median':
                spect_power_median[frequency] = np.median(spect_power_median_buffer[frequency])
            if interp_type == 'average':
                spect_power_median[frequency] = np.mean(spect_power_median_buffer[frequency])
            if interp_type == 'mask':
                spect_power_median[frequency] = 1
        # create clean power duplicate of power
        tf_power_clean = np.zeros([len(tf_power), len(tf_power[0]), len(tf_power[0][0])])
        for channel in range(0, len(tf_artifact_mask)):
            for time in range(0, len(tf_artifact_mask[channel])):
                for frequency in range(0, len(tf_power[channel])):
                    tf_power_clean[channel][frequency][time] = tf_power[channel][frequency][time]
        # replace bad data with median of good data for each frequency
        for channel in range(0, len(tf_artifact_mask)):
            for time in range(0, len(tf_artifact_mask[channel])):
                if tf_artifact_mask[channel][time] == 1:
                    for frequency in range(0, len(tf_power_clean[channel])):
                        tf_power_clean[channel][frequency][time] = spect_power_median[frequency]
    else:
##        print('CANNOT INTERPOLATE')
        tf_power_clean = tf_power
        
##    return tf_power
    return tf_power_clean



# RECODE 7/20/22 JG, interpolate_tf_power_singleband_across_channel INTERPOLATES SEPARATELY AT AT EACH BAND RANGE AVERAGE
# remove outliers from interpolated data: best to use AFTER extend/connect time artifacts and AFTER interpolation as final cleaning procedure
# input: tf_power_overtime[channel][freq][time], tf_artifact_mask[channel][time] where tf_artifact_mask has already been populated/cleaned
def reject_tf_power_outliers_interpolated_overtime(power_overtime, mask, outlier_std_multiplier):
    outlier_std_multiplier = 2.5
    # first get array of good data
    buffer_append = 0
    power_buffer_length = int((len(mask) * len(mask[0])) - np.sum(mask))
    power_buffer_clean = np.zeros(power_buffer_length)
    for channel in range(0, len(mask)):
        for time in range(0, len(mask[channel])):
            if mask[channel][time] == 0:
                power_buffer_clean[buffer_append] = power_overtime[channel][time]
                buffer_append += 1
    # calculate median across channels over time
    power_median_clean = np.median(power_buffer_clean)
    power_std_clean = np.std(power_buffer_clean) * outlier_std_multiplier
    # create clean mask duplicate of mask
    mask_clean = np.zeros([len(mask), len(mask[0])])
    for channel in range(0, len(mask)):
        for time in range(0, len(mask[channel])):
            if mask[channel][time] == 1:
                mask_clean[channel][time] = 1
    # create clean power duplicate
    power_overtime_clean = np.zeros([len(power_overtime), len(power_overtime[0])])
    for channel in range(0, len(power_overtime_clean)):
        for time in range(0, len(power_overtime_clean[channel])):
            power_overtime_clean[channel][time] = power_overtime[channel][time]
    # finally, replace 0s with 1s for values outside median/std
    for channel in range(0, len(mask)):
        for time in range(0, len(mask[channel])):
            if power_overtime[channel][time] > (power_median_clean + power_std_clean):
                power_overtime_clean[channel][time] = power_median_clean
                mask_clean[channel][time] = 1
##                print('CHANGING ch: ' + str(channel) + ', time: ' + str(time) + ', amp: '
##                      + str(np.round(power_overtime[channel][time], 4)) + ', thresh: '
##                      + str(power_median_clean + power_std_clean))

    return power_overtime_clean, mask_clean









