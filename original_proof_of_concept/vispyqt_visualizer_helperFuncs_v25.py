# vispyqt visualizer helper funcs 2/28/23, ported from helperFuncs_pysimplegui_betaV4.py located: C:\Users\James\Documents\Github\Petal\programs\processing_refactor
import sys
import math

import scipy
import numpy as np
import pandas as pd


from datetime import datetime
from scipy.stats import sem
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from scipy.signal import welch
from scipy.integrate import simps
from scipy.signal import spectrogram
from mne.time_frequency import psd_array_multitaper, tfr_array_multitaper
from mne.preprocessing import EOGRegression
import mne

from scipy.signal import lfilter, lfilter_zi
from mne.filter import create_filter

#JG TEST THE ERROR: https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=sys.maxsize)











def getFFT(data, sampleRate):                                                   # Data should be timeseries of single channel
    fftArray = np.fft.fft(data/len(data))                                       # Gives complex numbers, including imaginary "phase" component
    amplitudeArray = abs(fftArray)                                              # Abs of complex number is amplitude component, gives two-sided frequency array of positive values
    amplitudeArrayRange = np.arange(len(amplitudeArray))                        # make array from 0 to length of data
    epochNumber = len(fftArray)/sampleRate                                      # divide each value by how many seconds of data
    totalFrequency = amplitudeArrayRange/epochNumber                            # both-sided frequency range
    positiveFrequency = totalFrequency[range(int(len(amplitudeArray)/2))]       # one side frequency range of frequency
    positiveHertzValues = amplitudeArray[range(int(len(amplitudeArray)/2))]     # one side array of hz power values

##    return np.array([positiveHertzValues, positiveFrequency])
    return positiveHertzValues, positiveFrequency


### input: data[channel][time]
### output: fft_values[channel][value], fft_freq[freq] where len(fft_freq) = len(fft_values[channel]) = int(len(data[0]) / 2)
##def get_FFT_multichan(data, sampling_rate):                                              # Data should be timeseries of single channel
##    fft_freq = np.zeros(int(len(data[0]) / 2))
##    fft_values = np.zeros([len(data), int(len(data[0]) / 2)])
##    for channel in range(0, len(data)):
##        fftArray = np.fft.fft(data[0]/len(data[0]))                                       # Gives complex numbers, including imaginary "phase" component
##        amplitudeArray = abs(fftArray)                                              # Abs of complex number is amplitude component, gives two-sided frequency array of positive values
##        amplitudeArrayRange = np.arange(len(amplitudeArray))                        # make array from 0 to length of data
##        epochNumber = len(fftArray)/sampling_rate                                      # divide each value by how many seconds of data
##        totalFrequency = amplitudeArrayRange/epochNumber                            # both-sided frequency range
##        positiveFrequency = totalFrequency[range(int(len(amplitudeArray)/2))]       # one side frequency range of frequency
##        positiveHertzValues = amplitudeArray[range(int(len(amplitudeArray)/2))]     # one side array of hz power values
##        fft_values[channel] = positiveHertzValues
##    fft_freq = positiveFrequency
##
##    return fft_values, fft_freq








# input: data[channel][sample], example bandRange = np.array([[1, 3], [3, 8], [8, 13], [13, 30]])
# output: bandpower[channel][band]
def getBandpower_2023(data, nperseg, nfft, samplingRate, bandRange):
    numSensors = len(data)
    bandPower = np.zeros([numSensors, len(bandRange)])
    for channel in range(0, len(data)):
        dataBuffer = data[channel]
        f, t, Sxx = scipy.signal.spectrogram(dataBuffer, fs=samplingRate, nperseg=nperseg, nfft=nfft)             # nfft is by default =nperseg
        for band in range(0, len(bandRange)):
            bandRangePosition = [(int(round((bandRange[band, 0])/f[1]))), (int(round((bandRange[band, 1])/f[1]))+1)]  # plus 1 because np.mean leaves out the final number
            bandPower[channel][band] = np.mean([np.asarray(Sxx[bandRangePosition[0]:bandRangePosition[1]])])

    return bandPower



# https://www.codegrepper.com/code-examples/python/find+the+closest+value+in+an+array+python
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]



# ====================== FILTERING ======================

def butter_highpass_filterAPI(data, lowcut, sampling_rate, highpassOrder):
    nyq = 0.5 * sampling_rate
    high = lowcut / nyq
    iir_numer, iir_denom = butter(highpassOrder, [high], btype='high')
    return lfilter(iir_numer, iir_denom, data)

def butter_lowpass_filterAPI(data, highcut, sampling_rate, lowpassOrder):
    nyq = 0.5 * sampling_rate
    low = highcut / nyq
    iir_numer, iir_denom = butter(lowpassOrder, [low], btype='low')
    return lfilter(iir_numer, iir_denom, data)

def butter_bandpass_filterAPI(data, lowcut, highcut, sampling_rate, bandpassOrder):
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    iir_numer, iir_denom = butter(bandpassOrder, [low, high], btype='band')
    return lfilter(iir_numer, iir_denom, data)

def mirror_highpass_filterAPI(data, numSensors, multiplier, lowcut, sampling_rate, highpassOrder):
    windowLength = len(data[0])
    multiplierHalf = int(round(multiplier/4))
    mirrorHighpassData = []
    for channel in range(0, len(data)):
        mirror_channel = np.tile(np.append(data[channel], np.flip(data[channel], -1)), int(multiplier / 2))
        highpassed_channel = butter_highpass_filterAPI(mirror_channel, lowcut, sampling_rate, highpassOrder)
        lower_bound = len(data[channel]) * int(multiplier / 2)
        upper_bound = len(data[channel]) * (int(multiplier / 2) + 1)
        mirrorHighpassData.append(highpassed_channel[lower_bound : upper_bound])
##        mirrorHighpassData.append(highpassed_channel[(windowLength*multiplierHalf):(windowLength*(multiplierHalf+1))])

    return mirrorHighpassData
            
def mirror_lowpass_filterAPI(data, numSensors, multiplier, highcut, sampling_rate, lowpassOrder):
    windowLength = len(data[0])
    multiplierHalf = int(round(multiplier/4))
    mirrorLowpassData = []
    for channel in range(0, len(data)):
        mirror_channel = np.tile(np.append(data[channel], np.flip(data[channel], -1)), int(multiplier / 2))
        lowpassed_channel = butter_highpass_filterAPI(mirror_channel, highcut, sampling_rate, lowpassOrder)
        lower_bound = len(data[channel]) * int(multiplier / 2)
        upper_bound = len(data[channel]) * (int(multiplier / 2) + 1)
        mirrorLowpassData.append(lowpassed_channel[lower_bound : upper_bound])
##        mirrorLowpassData.append(lowpassed_channel[(windowLength*multiplierHalf):(windowLength*(multiplierHalf+1))])

    return mirrorLowpassData

def mirror_bandpass_filterAPI(data, numSensors, multiplier, lowcut, highcut, sampling_rate, bandpassOrder):
    windowLength = len(data[0])
    multiplierHalf = int(round(multiplier/4))
    mirrorBandpassData = []
    for channel in range(0, len(data)):
        mirror_channel = np.tile(np.append(data[channel], np.flip(data[channel], -1)), int(multiplier / 2))
        bandpassed_channel = butter_bandpass_filterAPI(mirror_channel, lowcut, highcut, sampling_rate, bandpassOrder)
        lower_bound = len(data[channel]) * int(multiplier / 2)
        upper_bound = len(data[channel]) * (int(multiplier / 2) + 1)
        mirrorBandpassData.append(bandpassed_channel[lower_bound : upper_bound])
##        mirrorBandpassData.append(bandpassed_channel[(windowLength*multiplierHalf):(windowLength*(multiplierHalf+1))])

    return mirrorBandpassData






# ======= TIME FREQUENCY

# input: data[channel][time]
# output: dataPSD[epoch][channel][freq][time] where epoch always = 1
def getMultitaper(data, freqs, sampling_rate):
    dataBufferEpoch = np.zeros([len(data), len(data[0])])
    epochArrayPSD = np.zeros([1, len(data), len(freqs), len(data[0])])
    for channel in range(0, len(data)):                                                            # calculate psd for each channel
        print('Calculating psd for channel ' + str(channel) + '...')
        dataBufferEpoch[channel] = np.asarray(data[channel])
        # note: len(psd[0][0][0]) = time = len(data[channel]) = len(lsl_array)
        psd = tfr_array_multitaper(np.array([[dataBufferEpoch[channel]]]), sampling_rate, freqs=freqs, output='power')
        epochArrayPSD[0][channel] = psd[0][0]

    return epochArrayPSD

# input: data[time], new_sampling_rate where data are downsampled
# output: dataPSD[epoch][channel][freq][time] where epoch and channel always = 1
def getMultitaper_singlechan(data, freqs, new_sampling_rate):
    dataBufferEpoch = np.zeros([1, len(data)])
    epochArrayPSD = np.zeros([1, 1, len(freqs), len(data)])
    print('Calculating psd for single channel')
    dataBufferEpoch[0] = np.asarray(data)
    psd = tfr_array_multitaper(np.array([[dataBufferEpoch[0]]]), new_sampling_rate, freqs=freqs, output='power')
##    psd = tfr_array_multitaper(np.array([[dataBufferEpoch[0]]]), int(samplingRate / 4), freqs=freqs, output='power')
    epochArrayPSD[0][0] = psd[0][0]

    return epochArrayPSD





























# =============================== ARTIFACT REJECT

# return 2D artifact_time, artifact_mask
# artifact_time[channel][start1, end1, …]
# artifact_mask[channel][sample] with 1/0s
# strict == 'off' default, strict == 'on' means strict start/end artifact times, good for time frequency
def artifact_detect_moving_window(data, win, step, threshold, strict):
    artifact_time = [[], [], [], []]
    for channel in range(0, len(data)):
        ongoing = 0
        last_bound_reject = -1
        artifact_time_buffer = []
        bound_range = np.array([0, win])
        bound_reject = np.array([0, 0])
        total_detections = math.floor((len(data[channel]) - win) / step) - 1
        #begin detections
        if total_detections > 2:
            for this_detection in range(0, total_detections - 1):
                this_mean = np.mean(data[channel][bound_range[0]:bound_range[1]])
                next_mean = np.mean(data[channel][bound_range[0] + win:bound_range[1] + win])
                if abs(this_mean - next_mean) >= threshold:                                     # If an artifact is detected
                    if ongoing == 0:                                                            # If this is the start of a new artifact window within a channel
                        bound_reject[0] = bound_range[0]                                        # Define start of artifact window range as boundReject[0]
                        ongoing = 1
                bound_range = bound_range + step
                if abs(this_mean - next_mean) < threshold and ongoing == 1:                     # If this is the end of an ongoing artifact
                    bound_reject[1] = bound_range[1] - step                                     # Define end of artifact window
                    if bound_reject[0] <= last_bound_reject:
                        artifact_time_buffer[len(artifact_time_buffer) - 1] = bound_range[1]
                    else:
                        artifact_time_buffer.append(bound_reject[0])
                        artifact_time_buffer.append(bound_reject[1])
                    last_bound_reject = bound_reject[1] + step
                    ongoing = 0
                #last detection
                if this_detection == total_detections - 2:                                      # If this is the last test
                    if abs(this_mean - next_mean) >= threshold and ongoing == 1:                # And there is still ongoing artifact, need to fill variables
                        bound_reject[1] = bound_range[1] - step                                 # Define end of artifact window
                        if bound_reject[0] <= last_bound_reject:
                            artifact_time_buffer[len(artifact_time_buffer) - 1] = bound_range[1]
                        else:
                            artifact_time_buffer.append(bound_reject[0])
                            artifact_time_buffer.append(bound_reject[1])

        # catch final edge case
        this_mean = np.mean(data[channel][len(data[channel]) - (win * 2) - 1:len(data[channel]) - win - 1])
        next_mean = np.mean(data[channel][len(data[channel]) - win - 1:len(data[channel]) - 1])
        if abs(this_mean - next_mean) >= threshold:
            bound_reject[1] = len(data[channel]) - 1
            if ongoing == 1:
                artifact_time_buffer[len(artifact_time_buffer) - 1] = len(data[channel]) - 1
            else:
                if bound_reject[0] <= last_bound_reject:
                    artifact_time_buffer[len(artifact_time_buffer) - 1] = len(data[channel]) - 1
                else:
                    artifact_time_buffer.append(bound_reject[0])
                    artifact_time_buffer.append(len(data[channel]) - 1)

######        # STRICT ONLY: CREATE STRICT ARTIFACT TIME ARRAY
######        if strict == 'on':
######            if len(artifact_time_buffer) > 0:
######                for value in range(0, len(artifact_time_buffer), 2):
######                    artifact_time_buffer[value] = artifact_time_buffer[value] + win - step
######                    artifact_time_buffer[value + 1] = artifact_time_buffer[value + 1] - win + step

        # catch first edge case, define 0 sample if artifact start is <= win + step
        if len(artifact_time_buffer) > 0:
            if artifact_time_buffer[0] <= win + step:
                artifact_time_buffer[0] = 0
##        # catch last edge case, define len(data[0]) sample if artifact end is >= (len(data[0]) - win - step
            if artifact_time_buffer[len(artifact_time_buffer) - 1] >= (len(data[channel]) - win - step):
                artifact_time_buffer[len(artifact_time_buffer) - 1] = len(data[channel]) - 1

        # fill final artifact_time array
        artifact_time[channel] = artifact_time_buffer

    # finally after run full loop, make artifact_mask
    artifact_mask = np.zeros([len(data), len(data[0])])
    for channel in range(0, len(data)):
        if len(artifact_time[channel]) > 0:
            ongoing = 0
            for sample in range(0, len(data[0])):
                if sample in artifact_time[channel]:
                    if ongoing == 0:
                        ongoing = 1
                    else:
                        ongoing = 0
                if ongoing == 1:
                    artifact_mask[channel][sample] = 1
            # catch last case
            if artifact_mask[channel][len(artifact_mask[channel]) - 2] == 1:
                artifact_mask[channel][len(artifact_mask[channel]) - 1] == 1

    return artifact_time, artifact_mask











### TAKEN FROM EEGSleepHelper.PY 5/22/23: C:\Users\James\Documents\Github\Petal\OcularCorrect_MNE\sleep\code
##def artifactHEOG_Detect_New(data, numChannels, corWindow, corThreshold, stepOverlap, stringCount):
##    averageHeogVoltage = 0
##    runningHeogVoltage = np.zeros([stringCount])
##    totalHeogTime = 0
##    totalNumberOfHeog = 0
##    totalAverageCor = 0
##    dataFP1 = data[1]
##    dataFP2 = data[2]
####    dataFP1 = data[1]-np.mean(data[1])
####    dataFP2 = data[2]-np.mean(data[2])
##    runningCor = np.zeros([stringCount])
##    numberOfDetections = math.floor((len(dataFP1)-corWindow)/stepOverlap)-1
##    boundRange = np.array([0, corWindow])
####    print('=================')
##    if numberOfDetections > 2:
##        for i in range(0, numberOfDetections-1):
##            dataWindowFP1 = dataFP1[boundRange[0]:boundRange[1]]
##            dataWindowFP2 = dataFP2[boundRange[0]:boundRange[1]]
##            mean_x = np.mean( dataWindowFP1 )
##            mean_y = np.mean( dataWindowFP2 )
##            std_x  = np.std ( dataWindowFP1 )
##            std_y  = np.std ( dataWindowFP2 )
##            n      = len    ( dataWindowFP1 )
##            thisCor = np.correlate( dataWindowFP1 - mean_x, dataWindowFP2 - mean_y, mode = 'valid' )[0] / n / ( std_x * std_y )
##            thisVolt = (abs(dataFP1[boundRange[0]]-dataFP1[boundRange[1]]) + abs(dataFP2[boundRange[0]]-dataFP2[boundRange[1]]))/2
##
##            #if values are unchanging in the boundrange window, definitely bad data and mark with a cor of 0 as an artifact
##            if str(thisCor) == 'nan':
##                thisCor = 0
##            
##            for corr in range(0, len(runningCor)-1):
##                runningCor[corr] = runningCor[corr+1]
##                runningHeogVoltage[corr] = runningHeogVoltage[corr+1]
##            runningCor[len(runningCor)-1] = thisCor
##            runningHeogVoltage[len(runningHeogVoltage)-1] = thisVolt
##
##            if np.mean(runningCor) < corThreshold*(-1):
##                totalNumberOfHeog += 1
##                totalAverageCor += np.mean(runningCor)
##                averageHeogVoltage += np.mean(runningHeogVoltage)
##                totalHeogTime += boundRange[1]-(boundRange[0]-(stepOverlap*2))
##                averageHeogTime = ((boundRange[0]-(stepOverlap*2))+boundRange[1])/2
####                print('COR DETECTED: average time at: ' + str(averageHeogTime) + ', voltage: ' + str(np.mean(runningHeogVoltage)))
####                print('BIG COR from: ' + str(boundRange[0]-(stepOverlap*2)) + ', ' + str(boundRange[1]))
####                print('RunningCor is: ' + str(runningCor))
####                print('np.mean(RunningCor) is: ' + str(np.mean(runningCor)))
##            boundRange = boundRange + stepOverlap
##
##    if totalNumberOfHeog > 0:
##        averageHeogVoltage = averageHeogVoltage/totalNumberOfHeog
##        totalAverageCor = totalAverageCor/totalNumberOfHeog
##    else:
##        averageHeogVoltage = 0
##        totalAverageCor = 0
##
##    finalReturn = np.zeros(4)
##    for value in range(0, len(finalReturn)):
##        finalReturn[0] = totalNumberOfHeog
##        finalReturn[1] = totalHeogTime
##        finalReturn[2] = averageHeogVoltage
##        finalReturn[3] = totalAverageCor
##
##    #replace nan with 0 in finalReturn
##    for value in range(0, len(finalReturn)):
##        if finalReturn[value] != finalReturn[value]:
##            finalReturn[value] = 0
##
##    return finalReturn
####    return totalNumberOfHeog












# ============= ARTIFACT HELPERS JULY 25 2022

# 1 channel get artifact time from artifact mask
def get_artifact_time_from_mask(artifact_mask):
    artifact_time = []
    ongoing = 0
    for sample in range(0, len(artifact_mask)):
        if artifact_mask[sample] == 1:
            if ongoing == 0:
                artifact_time.append(sample)
                ongoing = 1
        if artifact_mask[sample] == 0:
            if ongoing == 1:
                artifact_time.append(sample - 1)
                ongoing = 0
        # check last case
        if sample == len(artifact_mask) - 1:
            if ongoing == 1:
                artifact_time.append(sample)

    return artifact_time

# ==================== TIME FREQUENCY

#### ========== ARTIFACT MASK FUNCTIONS 7/10/22

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

# extend each artifact_time = 1 with a +/- 1 to catch edge cases: iterate, flip, iterate, flip
# input: artifact_mask[channel][time], extend_length = +/- extend_length samples around artifacts
# output: same as input with +/- 1 time buffer
def extend_artifact_mask(artifact_mask, extend_length):
    # connect_length = 0            # default for time frequency
    # connect_length = 5            # default for time series by sample
    # forward check
    for channel in range(0, len(artifact_mask)):
        for time in range(extend_length, len(artifact_mask[channel])):
            if artifact_mask[channel][time] == 1:
                artifact_mask[channel][time - extend_length:time] = 1
    # backward check
    artifact_mask = np.flip(artifact_mask, 1)
    for channel in range(0, len(artifact_mask)):
        for time in range(extend_length, len(artifact_mask[channel])):
            if artifact_mask[channel][time] == 1:
                artifact_mask[channel][time - extend_length:time] = 1
    # flip back for final buffered artifact array
    artifact_mask = np.flip(artifact_mask, 1)

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

# input: artifact_mask[channel][sample]
# output: artifactTime[ch1[start, end, start, end,...],ch2[start, end,...]...]
def get_artifact_time_from_mask(artifact_mask):
    artifact_time_output = []
    for channel in range(0, len(artifact_mask)):
        ongoing = 0
        artifact_time = []
        for sample in range(0, len(artifact_mask[channel])):
            if artifact_mask[channel][sample] == 1:
                if ongoing == 0:
                    artifact_time.append(sample)
                    ongoing = 1
            if artifact_mask[channel][sample] == 0:
                if ongoing == 1:
                    artifact_time.append(sample)
##                    artifact_time.append(sample - 1)
                    ongoing = 0
            # check last case
            if sample == len(artifact_mask[channel]) - 1:
                if ongoing == 1:
                    artifact_time.append(sample)
        artifact_time_output.append(artifact_time)

    return artifact_time_output





############# 2022 code

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







# =============== TIME FREQUENCY 2023
# === TF SCIPY SPECTRAL POWER
# input: raw_data_eeg[channel][time], raw_lsl_eeg[time]
# output: tf_power_trim_interp[channel][frequency][time], tf_freq_trim[frequency], tf_time_lsl[time], tf_artifact_mask[channel][time]
# output: tf_time_lsl[time], tf_artifact_mask_clean[time] where len(time) == len(tf_time) and tf_time_lsl is converted to lsl times and tf_artifact_mask has 0/1 for interpolated datapoints
def get_spectral_interp_2023(raw_data_eeg, raw_lsl_eeg, band_range, sampling_rate, nfft, nperseg, interp_type, output_type, interpolation_toggle, artifactTime):
####    output_type = 'raw'             # options: raw, bandpower
####    interpolation_toggle = 'on'     # options: on, off
##    interp_type = 'off'        # options: 'median', 'mean', 'mask'
    # === SPECT: scipy.signal.spectrogram (scaling='density' default, or 'spectrum'), input[channel][time], output[channel][freq][time]
    total_freq = [np.min(band_range), np.max(band_range)]
    tf_freq, tf_time, tf_power = scipy.signal.spectrogram(raw_data_eeg, fs=sampling_rate, nperseg=nperseg, nfft=nfft) # scaling='density' is default, or 'spectrum'
##    tf_power = 10*np.log10(tf_power)        # === optional scale to DB using 10*np.log10(data), sometimes produces nan, may need to remove them
    tf_power_trim, tf_freq_trim = get_time_frequency_singleband(tf_power, tf_freq, total_freq)              # === trim tf_power, tf_freq to frequencies of interest to limit computing power
    # convert time to lsl for tf_power_trim
    tf_time_samples = tf_time * sampling_rate               # convert spect_time to samples
    tf_time_lsl = np.zeros(len(tf_time_samples))            # optional: convert spect_time to lsl
    for value in range(0, len(tf_time_samples)):
        tf_time_lsl[value] = raw_lsl_eeg[int(tf_time_samples[value])]

    # ====== Interpolation basic variables
    if interpolation_toggle == 'on':
##        # artifact reject raw eeg data time series
##        artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg = 32, 8, 25
##        mirror_bandpass_data_artifact_eeg = mirror_bandpass_filterAPI(raw_data_eeg_original, len(raw_data_eeg), 8, .01, 30, sampling_rate, 2)
##        artifactTime, artifactMask = artifact_detect_moving_window(mirror_bandpass_data_artifact_eeg, artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg, 'on')
        # get tf_artifact_mask for interp vars, interpolate ACROSS CHANNEL tf_power_trim[channel][frequency][time] using tf_artifact_mask[channel][time]
        tf_artifact_mask = get_artifact_mask(artifactTime, tf_time_samples)
        tf_artifact_mask = connect_artifact_mask(tf_artifact_mask, 1)                # 1 = connect artifact times +/- 1
        tf_artifact_mask = reject_channel_artifact_mask(tf_artifact_mask, 0.5)       # 0.5 = 50% percent good data
        tf_power_trim_interp = interpolate_tf_power_singleband_across_channel(tf_power_trim, tf_artifact_mask, str(interp_type))

    # ====== Bandpower tree
    if output_type == 'bandpower':

        # ====== Raw tree
        if interpolation_toggle == 'off':
            # === TF RAW: bandpower OVERTIME from tf_power_trim
            tf_bandpower_trim_raw_overtime = np.zeros([len(band_range), len(raw_data_eeg), len(tf_time)])
            for band in range(0, len(band_range)):
                tf_bandpower_trim_raw_overtime_buffer = get_time_frequency_overtime(tf_power_trim, tf_freq_trim, band_range[band])
                for channel in range(0, len(raw_data_eeg)):
                    tf_bandpower_trim_raw_overtime[band][channel] = tf_bandpower_trim_raw_overtime_buffer[channel]
            # === TF RAW: bandpower AVETIME from tf_power_trim
            tf_bandpower_raw_avetime = np.zeros([len(band_range), len(raw_data_eeg)])
            for band in range(0, len(band_range)):
                for channel in range(0, len(raw_data_eeg)):
                    tf_bandpower_raw_avetime[band][channel] = np.mean(tf_bandpower_trim_raw_overtime[band][channel])
        # ====== Interpolation tree
        if interpolation_toggle == 'on':
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
            # === TF INTERP: bandpower OVERTIME from tf_power_trim
            tf_bandpower_interp_avetime_clean = np.zeros([len(band_range), len(raw_data_eeg)])
            for band in range(0, len(band_range)):
                for channel in range(0, len(raw_data_eeg)):
                    tf_bandpower_interp_avetime_clean[band][channel] = np.mean(tf_bandpower_interp_overtime_clean[band][channel])

    # output
    if output_type == 'raw' and interpolation_toggle == 'off':
        return tf_power_trim, tf_freq_trim, tf_time_lsl
    if output_type == 'raw' and interpolation_toggle == 'on':
        return tf_power_trim_interp, tf_freq_trim, tf_time_lsl, tf_artifact_mask
    if output_type == 'bandpower' and interpolation_toggle == 'off':
        return tf_bandpower_raw_avetime, tf_bandpower_trim_raw_overtime, tf_time_lsl
    if output_type == 'bandpower' and interpolation_toggle == 'on':
        return tf_bandpower_interp_avetime_clean, tf_bandpower_interp_overtime_clean, tf_time_lsl, tf_artifact_mask_clean



# === TF SCIPY SPECTRAL POWER
# input: raw_data_eeg[channel][time], raw_lsl_eeg[time]
# output: tf_power_trim_interp[channel][frequency][time], tf_freq_trim[frequency], tf_time_lsl[time], tf_artifact_mask[channel][time]
# output: tf_time_lsl[time], tf_artifact_mask_clean[time] where len(time) == len(tf_time) and tf_time_lsl is converted to lsl times and tf_artifact_mask has 0/1 for interpolated datapoints
def get_spectral_power_2023(raw_data_eeg, raw_lsl_eeg, band_range, sampling_rate, nfft, nperseg, output_type):
####    output_type = 'raw'             # options: raw, bandpower
    # === SPECT: scipy.signal.spectrogram (scaling='density' default, or 'spectrum'), input[channel][time], output[channel][freq][time]
    total_freq = [np.min(band_range), np.max(band_range)]
    tf_freq, tf_time, tf_power = scipy.signal.spectrogram(raw_data_eeg, fs=sampling_rate, nperseg=nperseg, nfft=nfft) # scaling='density' is default, or 'spectrum'
##    tf_power = 10*np.log10(tf_power)        # === optional scale to DB using 10*np.log10(data), sometimes produces nan, may need to remove them
    tf_power_trim, tf_freq_trim = get_time_frequency_singleband(tf_power, tf_freq, total_freq)              # === trim tf_power, tf_freq to frequencies of interest to limit computing power
    # convert time to lsl for tf_power_trim
    tf_time_samples = tf_time * sampling_rate               # convert spect_time to samples
    tf_time_lsl = np.zeros(len(tf_time_samples))            # optional: convert spect_time to lsl
    for value in range(0, len(tf_time_samples)):
        tf_time_lsl[value] = raw_lsl_eeg[int(tf_time_samples[value])]

    # ====== Bandpower tree
    if output_type == 'bandpower':
##        # ====== Raw tree
##        if interpolation_toggle == 'off':
        # === TF RAW: bandpower OVERTIME from tf_power_trim
        tf_bandpower_trim_raw_overtime = np.zeros([len(band_range), len(raw_data_eeg), len(tf_time)])
        for band in range(0, len(band_range)):
            tf_bandpower_trim_raw_overtime_buffer = get_time_frequency_overtime(tf_power_trim, tf_freq_trim, band_range[band])
            for channel in range(0, len(raw_data_eeg)):
                tf_bandpower_trim_raw_overtime[band][channel] = tf_bandpower_trim_raw_overtime_buffer[channel]
        # === TF RAW: bandpower AVETIME from tf_power_trim
        tf_bandpower_raw_avetime = np.zeros([len(band_range), len(raw_data_eeg)])
        for band in range(0, len(band_range)):
            for channel in range(0, len(raw_data_eeg)):
                tf_bandpower_raw_avetime[band][channel] = np.mean(tf_bandpower_trim_raw_overtime[band][channel])

    # output
    if output_type == 'raw':
        return tf_power_trim, tf_freq_trim, tf_time_lsl
    if output_type == 'bandpower':
        return tf_bandpower_raw_avetime, tf_bandpower_trim_raw_overtime, tf_time_lsl

















# ============== TIME FREQUENCY FUNCTIONS FINAL 11/30/22 START

### === TF SCIPY SPECTRAL POWER
### input: raw_data_eeg[channel][time], raw_lsl_eeg[time]
### output: tf_power_trim_interp[channel][frequency][time], tf_freq_trim[frequency], tf_time_lsl[time], tf_artifact_mask[channel][time]
### output: tf_time_lsl[time], tf_artifact_mask_clean[time] where len(time) == len(tf_time) and tf_time_lsl is converted to lsl times and tf_artifact_mask has 0/1 for interpolated datapoints
##def get_spectral_interp_2022(raw_data_eeg, raw_lsl_eeg, raw_data_eeg_original, band_range, sampling_rate, nfft, nperseg, interp_type, output_type, interpolation_toggle):
######    output_type = 'raw'             # options: raw, bandpower
######    interpolation_toggle = 'on'     # options: on, off
####    interp_type = 'off'        # options: 'median', 'mean', 'mask'
##    # === SPECT: scipy.signal.spectrogram (scaling='density' default, or 'spectrum'), input[channel][time], output[channel][freq][time]
##    total_freq = [np.min(band_range), np.max(band_range)]
##    tf_freq, tf_time, tf_power = scipy.signal.spectrogram(raw_data_eeg, fs=sampling_rate, nperseg=nperseg, nfft=nfft) # scaling='density' is default, or 'spectrum'
####    tf_power = 10*np.log10(tf_power)        # === optional scale to DB using 10*np.log10(data), sometimes produces nan, may need to remove them
##    tf_power_trim, tf_freq_trim = get_time_frequency_singleband(tf_power, tf_freq, total_freq)              # === trim tf_power, tf_freq to frequencies of interest to limit computing power
##    # convert time to lsl for tf_power_trim
##    tf_time_samples = tf_time * sampling_rate               # convert spect_time to samples
##    tf_time_lsl = np.zeros(len(tf_time_samples))            # optional: convert spect_time to lsl
##    for value in range(0, len(tf_time_samples)):
##        tf_time_lsl[value] = raw_lsl_eeg[int(tf_time_samples[value])]
##
##    # ====== Interpolation basic variables
##    if interpolation_toggle == 'on':
##        # artifact reject raw eeg data time series
##        artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg = 32, 8, 25
##        mirror_bandpass_data_artifact_eeg = mirror_bandpass_filterAPI(raw_data_eeg_original, len(raw_data_eeg), 8, .01, 30, sampling_rate, 2)
##        artifactTime, artifactMask = artifact_detect_moving_window(mirror_bandpass_data_artifact_eeg, artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg, 'on')
##        # get tf_artifact_mask for interp vars, interpolate ACROSS CHANNEL tf_power_trim[channel][frequency][time] using tf_artifact_mask[channel][time]
##        tf_artifact_mask = get_artifact_mask(artifactTime, tf_time_samples)
##        tf_artifact_mask = connect_artifact_mask(tf_artifact_mask, 1)                # 1 = connect artifact times +/- 1
##        tf_artifact_mask = reject_channel_artifact_mask(tf_artifact_mask, 0.5)       # 0.5 = 50% percent good data
##        tf_power_trim_interp = interpolate_tf_power_singleband_across_channel(tf_power_trim, tf_artifact_mask, str(interp_type))
##
####        # UPDATED 2023
####        tf_artifact_mask = artifactMask
####        tf_artifact_mask = extend_artifact_mask(tf_artifact_mask, 24)                 # see also 32: liberal psd paramter: 256/32 = 8 = 31.25 ms = extend artifact times +/- 8
####        tf_artifact_mask = connect_artifact_mask(tf_artifact_mask, 16)                # see also 8: liberal psd parameter: 256/16 = 16 = 62.5 ms = connect artifact times separated by +/- 16
####        # WITHIN CHANNEL
####        tf_power_trim_interp = interpolate_tf_power_singleband_within_channel(tf_power_trim, tf_artifact_mask, str(interp_type))
####        # ACROSS CHANNEL
####        tf_power_trim_interp = interpolate_tf_power_singleband_across_channel(tf_power_trim, tf_artifact_mask, str(interp_type))
##
##    # ====== Bandpower tree
##    if output_type == 'bandpower':
##        # ====== Raw tree
##        if interpolation_toggle == 'off':
##            # === TF RAW: bandpower OVERTIME from tf_power_trim
##            tf_bandpower_trim_raw_overtime = np.zeros([len(band_range), len(raw_data_eeg), len(tf_time)])
##            for band in range(0, len(band_range)):
##                tf_bandpower_trim_raw_overtime_buffer = get_time_frequency_overtime(tf_power_trim, tf_freq_trim, band_range[band])
##                for channel in range(0, len(raw_data_eeg)):
##                    tf_bandpower_trim_raw_overtime[band][channel] = tf_bandpower_trim_raw_overtime_buffer[channel]
##            # === TF RAW: bandpower AVETIME from tf_power_trim
##            tf_bandpower_raw_avetime = np.zeros([len(band_range), len(raw_data_eeg)])
##            for band in range(0, len(band_range)):
##                for channel in range(0, len(raw_data_eeg)):
##                    tf_bandpower_raw_avetime[band][channel] = np.mean(tf_bandpower_trim_raw_overtime[band][channel])
##        # ====== Interpolation tree
##        if interpolation_toggle == 'on':
##            tf_bandpower_interp_overtime_clean = np.zeros([len(band_range), len(raw_data_eeg), len(tf_time)])
##            # === TF INTERP: bandpower OVERTIME from tf_power_trim
##            for band in range(0, len(band_range)):
##                # calculate tf_power_trim_interp overtime averaged across band range frequencies
##                tf_power_trim_interp_overtime = get_time_frequency_overtime(tf_power_trim_interp, tf_freq_trim, band_range[band])
##                # DOUBLE CHECK THAT THIS IS UPDATING BUT USING THE OLDER TF_ARTIFACT_MASK SO WE HAVE DIFFERENT MASKS FOR EACH BAND                
##                # trim tf_power_trim_overtime_interp for outliers for the averaged band range by creating a new tf_artifact_mask, can replace with "mask_clean" to test == tf_artifact_mask
##                tf_power_trim_overtime_interp_clean, tf_artifact_mask_clean = reject_tf_power_outliers_interpolated_overtime(tf_power_trim_interp_overtime, tf_artifact_mask, 2.5)
##                # append data to the final tf_bandpower_interp_overtime_clean array
##                for channel in range(0, len(raw_data_eeg)):
##                    tf_bandpower_interp_overtime_clean[band][channel] = tf_power_trim_overtime_interp_clean[channel]
##            # === TF INTERP: bandpower OVERTIME from tf_power_trim
##            tf_bandpower_interp_avetime_clean = np.zeros([len(band_range), len(raw_data_eeg)])
##            for band in range(0, len(band_range)):
##                for channel in range(0, len(raw_data_eeg)):
##                    tf_bandpower_interp_avetime_clean[band][channel] = np.mean(tf_bandpower_interp_overtime_clean[band][channel])
##
##    # output
##    if output_type == 'raw' and interpolation_toggle == 'off':
##        return tf_power_trim, tf_freq_trim, tf_time_lsl
##    if output_type == 'raw' and interpolation_toggle == 'on':
##        return tf_power_trim_interp, tf_freq_trim, tf_time_lsl, tf_artifact_mask
##    if output_type == 'bandpower' and interpolation_toggle == 'off':
##        return tf_bandpower_raw_avetime, tf_bandpower_trim_raw_overtime, tf_time_lsl
##    if output_type == 'bandpower' and interpolation_toggle == 'on':
##        return tf_bandpower_interp_avetime_clean, tf_bandpower_interp_overtime_clean, tf_time_lsl, tf_artifact_mask_clean



# === PSD overtime and avetime ARTIFACT REJECTION AND INTERPOLATION
# input: raw_data_eeg[channel][time], raw_lsl_eeg[time]
# NOTE: raw_data_eeg_original is used for artifact rejection only, raw_data_eeg may be filtered/processed/etc. and is used to calculate TF values
# output: psd_bandpower_interp_avetime_clean[band][channel], psd_bandpower_interp_overtime_clean[band][channel][time]
# output: psd_time_lsl[time], psd_artifact_mask_clean[time] where len(time) == len(psd_time) and psd_time_lsl is converted to lsl times and psd_artifact_mask has 0/1 for interpolated datapoints
def get_spectral_interp_psd_2022(raw_data_eeg, raw_lsl_eeg, raw_data_eeg_original, band_range, freqs, sampling_rate, interp_type, output_type, interpolation_toggle):
####    output_type = 'raw'             # options: raw, bandpower
####    interpolation_toggle = 'on'     # options: on, off
##    interp_type = 'off'        # options: 'median', 'mean', 'mask'
    # === MULTITAPER: tfr_array_multitaper, input[channel][time], output[epoch][channel][freq][time] where epoch = 1
    epochArrayPSD = getMultitaper(raw_data_eeg, freqs, sampling_rate)
##    total_freq = [np.min(band_range), np.max(band_range)]
    total_freq = [1, 30]
    psd_power_trim, psd_freq_trim = get_time_frequency_singleband(epochArrayPSD[0], freqs, total_freq)
##    psd_power_trim = np.sqrt(epochArrayPSD[0]) / 10               # optional: convert PSD using MNE_tfr_multitaper PSD to scipy.signal.spectrogram SpectralPower: np.sqrt(psd)/10

##    psd_power_trim, psd_freq_trim = epochArrayPSD[0], freqs
##    psd_power_trim_interp, psd_freq_trim = epochArrayPSD[0], freqs

    # ====== Interpolation basic variables
    if interpolation_toggle == 'on':
        # artifact reject raw eeg data time series
        artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg = 32, 8, 25            # see also: 64, 16, 75
        mirror_bandpass_data_artifact_eeg = mirror_bandpass_filterAPI(raw_data_eeg_original, len(raw_data_eeg), 8, .01, 30, sampling_rate, 2)
        artifactTime, artifactMask = artifact_detect_moving_window(mirror_bandpass_data_artifact_eeg, artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg, 'on')

        # get tf_artifact_mask for interp vars, interpolate WITHIN CHANNEL tf_power_trim[channel][frequency][time] using tf_artifact_mask[channel][time]
##        psd_artifact_mask = get_artifact_mask(artifactTime, raw_lsl_eeg)
        psd_artifact_mask = artifactMask
        psd_artifact_mask = extend_artifact_mask(psd_artifact_mask, 24)                 # see also 32: liberal psd paramter: 256/32 = 8 = 31.25 ms = extend artifact times +/- 8
        psd_artifact_mask = connect_artifact_mask(psd_artifact_mask, 16)                # see also 8: liberal psd parameter: 256/16 = 16 = 62.5 ms = connect artifact times separated by +/- 16
##        psd_artifact_mask = reject_channel_artifact_mask(psd_artifact_mask, 0.5)       # 0.5 = 50% percent good data
        psd_power_trim_interp = interpolate_tf_power_singleband_within_channel(psd_power_trim, psd_artifact_mask, str(interp_type))

    # ====== Bandpower tree
    if output_type == 'bandpower':
        # ====== Raw tree
        if interpolation_toggle == 'off':
            # === TF RAW: bandpower OVERTIME from tf_power_trim
            psd_bandpower_trim_raw_overtime = np.zeros([len(band_range), len(raw_data_eeg), len(raw_lsl_eeg)])
            for band in range(0, len(band_range)):
                psd_bandpower_trim_raw_overtime_buffer = get_time_frequency_overtime(psd_power_trim, psd_freq_trim, band_range[band])
                for channel in range(0, len(raw_data_eeg)):
                    psd_bandpower_trim_raw_overtime[band][channel] = psd_bandpower_trim_raw_overtime_buffer[channel]
            # === TF RAW: bandpower AVETIME from tf_power_trim
            psd_bandpower_raw_avetime = np.zeros([len(band_range), len(raw_data_eeg)])
            for band in range(0, len(band_range)):
                for channel in range(0, len(raw_data_eeg)):
                    psd_bandpower_raw_avetime[band][channel] = np.mean(psd_bandpower_trim_raw_overtime[band][channel])

        # ====== Interpolation tree
        if interpolation_toggle == 'on':
            psd_bandpower_interp_overtime_clean = np.zeros([len(band_range), len(raw_data_eeg), len(raw_lsl_eeg)])
            # === TF INTERP: bandpower OVERTIME from tf_power_trim
            for band in range(0, len(band_range)):
                # calculate tf_power_trim_interp overtime averaged across band range frequencies
                psd_power_trim_interp_overtime = get_time_frequency_overtime(psd_power_trim_interp, psd_freq_trim, band_range[band])
                psd_power_trim_overtime_interp_clean, psd_artifact_mask_clean = reject_tf_power_outliers_interpolated_overtime(psd_power_trim_interp_overtime, psd_artifact_mask, 2.5)
                # append data to the final tf_bandpower_interp_overtime_clean array
                for channel in range(0, len(raw_data_eeg)):
                    psd_bandpower_interp_overtime_clean[band][channel] = psd_power_trim_overtime_interp_clean[channel]
            # === TF INTERP: bandpower OVERTIME from tf_power_trim
            psd_bandpower_interp_avetime_clean = np.zeros([len(band_range), len(raw_data_eeg)])
            for band in range(0, len(band_range)):
                for channel in range(0, len(raw_data_eeg)):
                    psd_bandpower_interp_avetime_clean[band][channel] = np.mean(psd_bandpower_interp_overtime_clean[band][channel])
######
######    if interpolation_toggle == 'on':
######        print(' ==== artifactTime ==== ')
######        print(artifactTime)
######        print(len(artifactTime[0]))
######        print(len(artifactTime[1]))
######        print(len(artifactTime[2]))
######        print(len(artifactTime[3]))
######        print('artifactMask')
######        print(np.sum(artifactMask[0]))
######        print(np.sum(artifactMask[1]))
######        print(np.sum(artifactMask[2]))
######        print(np.sum(artifactMask[3]))
######        print('psd_artifact_mask')
######        print(np.sum(psd_artifact_mask[0]))
######        print(np.sum(psd_artifact_mask[1]))
######        print(np.sum(psd_artifact_mask[2]))
######        print(np.sum(psd_artifact_mask[3]))
######        print('===========================')
########        print(artifactMask[0])
######        print('-===========================')

    # output
    if output_type == 'raw' and interpolation_toggle == 'off':
        return psd_power_trim, psd_freq_trim, raw_lsl_eeg
    if output_type == 'raw' and interpolation_toggle == 'on':
        return psd_power_trim_interp, psd_freq_trim, raw_lsl_eeg, psd_artifact_mask
    if output_type == 'bandpower' and interpolation_toggle == 'off':
        return psd_bandpower_raw_avetime, psd_bandpower_trim_raw_overtime, raw_lsl_eeg
    if output_type == 'bandpower' and interpolation_toggle == 'on':
        return psd_bandpower_interp_avetime_clean, psd_bandpower_interp_overtime_clean, raw_lsl_eeg, psd_artifact_mask_clean


# ============== TIME FREQUENCY FUNCTIONS FINAL 11/30/22 END


# ================== FINAL TIME FREQUENCY INTERPOLATION HELPER FUNCS



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

# RECODE 11/27/22 JG, interpolate_tf_power_singleband_WITHIN_channel INTERPOLATES SEPARATELY AT EACH FREQUENCY WITHIN EACH CHANNEL SEPARATLEY
# description: corrects artifact flagged time points with median of good time points WITHIN channels separately for each FREQUENCY
# note: should work for singleband frequencies or bandpower averaged frequencies: just needs to be "overtime" with time component
# input: tf_artifact_mask[channel][artifact] where bad data artifact = 1 and good data = 0, interp_type: 'median', 'mean', 1 (note 1 is for masking in figures)
# output: tf_power with interpolated data
def interpolate_tf_power_singleband_within_channel(tf_power, tf_artifact_mask, interp_type):
    # median to subtract for each channel separately, spect_power_median[channel][frequency]
    spect_power_median = np.zeros([len(tf_artifact_mask), len(tf_power[0])])
    # check if all data are good or bad, if so then cannot interpolate
    artifact_sum_check = int(np.sum(tf_artifact_mask))
    artifact_length_check = int(len(tf_artifact_mask) * len(tf_artifact_mask[0]))
    if (artifact_sum_check != 0) and (artifact_sum_check != artifact_length_check):
        # get the median power of all good data for each frequency separately for interpolation
        for channel in range(0, len(tf_artifact_mask)):
            buffer_append = 0
            artifact_sum = int(np.sum(tf_artifact_mask[channel]))
            artifact_length = len(tf_artifact_mask[channel])
            spect_power_median_buffer = np.zeros([len(tf_power[0]), artifact_length - artifact_sum])
            for time in range(0, len(tf_artifact_mask[channel])):
                if tf_artifact_mask[channel][time] == 0:
                    for frequency in range(0, len(tf_power[channel])):
                        spect_power_median_buffer[frequency][buffer_append] = tf_power[channel][frequency][time]
##                        spect_power_median_buffer[frequency][buffer_append] = 1
                    buffer_append += 1
            # calculate the median at each frequency to account for log power?
            for frequency in range(0, len(spect_power_median[channel])):
##                spect_power_median[channel][frequency] = 1
##                spect_power_median[channel][frequency] = np.median(spect_power_median_buffer[frequency])
                if interp_type == 'median':
                    spect_power_median[channel][frequency] = np.median(spect_power_median_buffer[frequency])
                if interp_type == 'average':
                    spect_power_median[channel][frequency] = np.mean(spect_power_median_buffer[frequency])
                if interp_type == 'mask':
                    spect_power_median[channel][frequency] = 1
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
                        tf_power_clean[channel][frequency][time] = spect_power_median[channel][frequency]
    else:
##        print('CANNOT INTERPOLATE')
        tf_power_clean = tf_power

##    return tf_power
    return tf_power_clean




### input: data[channel][sample], mask[channel][sample] where 1 = artifact
##def interpolate_time_series(data, raw_data, sampling_rate, interp_type):
####    if device == 'accel':
####        artifact_win, artifact_step, artifact_thresh = 10, 2, 5
####        mirror_bandpass_data_artifact = mirror_bandpass_filterAPI(raw_data, len(raw_data), 8, 1, 15, sampling_rate, 2)
##    artifact_win, artifact_step, artifact_thresh = 32, 8, 25
##    mirror_bandpass_data_artifact = mirror_bandpass_filterAPI(raw_data, len(raw_data), 8, .01, 30, sampling_rate, 2)
##
####    artifact_time, artifact_mask = artifact_detect_moving_window(mirror_bandpass_data_artifact, artifact_win, artifact_step, artifact_thresh, 'on')
##    artifact_time, artifact_mask = artifact_detect_moving_window(mirror_bandpass_data_artifact, artifact_win, artifact_step, artifact_thresh, 'off')
##    # check if all data are good or bad, if so then cannot interpolate
##    data_median = np.zeros(len(data))
##    artifact_sum_check = int(np.sum(artifact_mask))
##    artifact_length_check = int(len(artifact_mask) * len(artifact_mask[0]))
##    if (artifact_sum_check != 0) and (artifact_sum_check != artifact_length_check):
##        for channel in range(0, len(artifact_mask)):
##            data_median_buffer = []
##            for time in range(0, len(artifact_mask[channel])):
##                if artifact_mask[channel][time] == 0:
##                    data_median_buffer.append(data[channel][time])
##            if interp_type == 'median':
##                data_median[channel] = np.median(data_median_buffer)
##            if interp_type == 'average':
##                data_median[channel] = np.mean(data_median_buffer)
##            if interp_type == 'mask':
##                data_median[channel] = 1
##        # create clean power duplicate of power
##        data_clean = np.zeros([len(data), len(data[0])])
##        for channel in range(0, len(data)):
##            for time in range(0, len(data[channel])):
##                data_clean[channel][time] = data[channel][time]
##        # replace bad data with median of good data for each frequency
##        for channel in range(0, len(artifact_mask)):
##            for time in range(0, len(artifact_mask[channel])):
##                if artifact_mask[channel][time] == 1:
##                    data_clean[channel][time] = data_median[channel]
##    else:
##        data_clean = data
##
##    return data_clean

















# input: data[channel][sample], mask[channel][sample] where 1 = artifact
def interpolate_time_series_2023(data, raw_data, sampling_rate, interp_type, artifact_time, artifact_mask):
##    artifact_win, artifact_step, artifact_thresh = 32, 8, 25
##    mirror_bandpass_data_artifact = mirror_bandpass_filterAPI(raw_data, len(raw_data), 8, .01, 30, sampling_rate, 2)
##    artifact_time, artifact_mask = artifact_detect_moving_window(mirror_bandpass_data_artifact, artifact_win, artifact_step, artifact_thresh, 'off')
    # check if all data are good or bad, if so then cannot interpolate
    data_median = np.zeros(len(data))
    artifact_sum_check = int(np.sum(artifact_mask))
    artifact_length_check = int(len(artifact_mask) * len(artifact_mask[0]))
    if (artifact_sum_check != 0) and (artifact_sum_check != artifact_length_check):
        for channel in range(0, len(artifact_mask)):
            data_median_buffer = []
            for time in range(0, len(artifact_mask[channel])):
                if artifact_mask[channel][time] == 0:
                    data_median_buffer.append(data[channel][time])
            if interp_type == 'median':
                data_median[channel] = np.median(data_median_buffer)
            if interp_type == 'average':
                data_median[channel] = np.mean(data_median_buffer)
            if interp_type == 'mask':
                data_median[channel] = 1
        # create clean power duplicate of power
        data_clean = np.zeros([len(data), len(data[0])])
        for channel in range(0, len(data)):
            for time in range(0, len(data[channel])):
                data_clean[channel][time] = data[channel][time]
        # replace bad data with median of good data for each frequency
        for channel in range(0, len(artifact_mask)):
            for time in range(0, len(artifact_mask[channel])):
                if artifact_mask[channel][time] == 1:
                    data_clean[channel][time] = data_median[channel]
    else:
        data_clean = data

    return data_clean











# ============================ 11/25/22 ALL PYSIMPLEGUI FUNCS ==========================


##### === Artifact Rejection EEG
##### input: raw_data_eeg[channel][sample], NOTE: raw_data_eeg is RAW DATA because we have CUSTOM FILTERS for artifact rejection
##### note: psd has full time points, therefore if we use strict = 'on' here may work less good than for overtime with far fewer timepoints
######def artifact_rejection_eeg(raw_data_eeg, channel_list_eeg, sampling_rate):
##### 2023 added toggle = 'low', 'mid', 'high'
####def artifact_rejection_eeg(raw_data_eeg, channel_list_eeg, sampling_rate, threshold):
####    if threshold == 'low':
####        artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg = 32, 8, 25
####    if threshold == 'mid':
####        artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg = 32, 8, 50
####    if threshold == 'high':
####        artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg = 32, 8, 100
######    artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg = 32, 8, 25
####    mirror_bandpass_data_artifact_eeg = mirror_bandpass_filterAPI(raw_data_eeg, len(raw_data_eeg), 8, .01, 30, sampling_rate, 2)
####    artifact_time_strict_eeg, artifact_mask_strict_eeg = artifact_detect_moving_window(mirror_bandpass_data_artifact_eeg, artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg, 'on')
####
####    return artifact_time_strict_eeg, artifact_mask_strict_eeg




# === Artifact Rejection EEG
# input: raw_data_eeg[channel][sample], NOTE: raw_data_eeg is RAW DATA because we have CUSTOM FILTERS for artifact rejection
# note: psd has full time points, therefore if we use strict = 'on' here may work less good than for overtime with far fewer timepoints
def artifact_rejection_eeg(raw_data_eeg, sampling_rate):
    # === EEG artifact reject: MOVING WINDOW GENERAL ARTIFACTS
##    getMultitaperBandPower(data, numSensors, band_range, freqs)
    artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg = 32, 8, 25
    mirror_bandpass_data_artifact_eeg = mirror_bandpass_filterAPI(raw_data_eeg, len(raw_data_eeg), 8, .01, 30, sampling_rate, 2)
    artifact_time_strict_eeg, artifact_mask_strict_eeg = artifact_detect_moving_window(mirror_bandpass_data_artifact_eeg, artifact_win_eeg, artifact_step_eeg, artifact_thresh_eeg, 'on')
##    save_artifact_rejection(str(filename_output), channel_list_eeg, artifact_time_strict_eeg)

    return artifact_time_strict_eeg, artifact_mask_strict_eeg

# === Artifact Rejection ACCEL NEW
# input: raw_data_eeg[channel][sample], NOTE: raw_data_eeg is RAW DATA because we have CUSTOM FILTERS for artifact rejection
def artifact_rejection_accel(raw_data_accel, sampling_rate):
    # === ACCEL artifact reject: MOVING WINDOW GENERAL ARTIFACTS
    artifact_win_accel, artifact_step_accel, artifact_thresh_accel = 16, 2, .1
##    artifact_win_accel, artifact_step_accel, artifact_thresh_accel = 8, 2, .1
##    artifact_win_accel, artifact_step_accel, artifact_thresh_accel = 48, 2, .1
    artifact_time_strict_accel, artifact_mask_strict_accel = artifact_detect_moving_window(raw_data_accel, artifact_win_accel, artifact_step_accel, artifact_thresh_accel, 'off')

##    mirror_bandpass_data_artifact_accel = mirror_bandpass_filterAPI(raw_data_accel, len(raw_data_accel), 8, 1, 15, sampling_rate, 2)
####    artifact_time_strict_accel, artifact_mask_strict_accel = artifact_detect_moving_window(mirror_bandpass_data_artifact_accel, artifact_win_accel, artifact_step_accel, artifact_thresh_accel, 'on')
##    artifact_time_strict_accel, artifact_mask_strict_accel = artifact_detect_moving_window(mirror_bandpass_data_artifact_accel, artifact_win_accel, artifact_step_accel, artifact_thresh_accel, 'off')

    return artifact_time_strict_accel, artifact_mask_strict_accel


# === Artifact Rejection GYRO NEW
# input: raw_data_eeg[channel][sample], NOTE: raw_data_eeg is RAW DATA because we have CUSTOM FILTERS for artifact rejection
def artifact_rejection_gyro(raw_data_gyro, sampling_rate):
    # === GYRO artifact reject: MOVING WINDOW GENERAL ARTIFACTS
    artifact_win_gyro, artifact_step_gyro, artifact_thresh_gyro = 16, 2, 10
##    artifact_win_gyro, artifact_step_gyro, artifact_thresh_gyro = 12, 2, 10
    artifact_time_strict_gyro, artifact_mask_strict_gyro = artifact_detect_moving_window(raw_data_gyro, artifact_win_gyro, artifact_step_gyro, artifact_thresh_gyro, 'off')

##    mirror_bandpass_data_artifact_gyro = mirror_bandpass_filterAPI(raw_data_gyro, len(raw_data_gyro), 8, 1, 15, sampling_rate, 2)
####    artifact_time_strict_gyro, artifact_mask_strict_gyro = artifact_detect_moving_window(mirror_bandpass_data_artifact_gyro, artifact_win_gyro, artifact_step_gyro, artifact_thresh_gyro, 'on')
##    artifact_time_strict_gyro, artifact_mask_strict_gyro = artifact_detect_moving_window(mirror_bandpass_data_artifact_gyro, artifact_win_gyro, artifact_step_gyro, artifact_thresh_gyro, 'off')

    return artifact_time_strict_gyro, artifact_mask_strict_gyro












####### === Artifact Rejection ACCEL OLD
####### input: raw_data_eeg[channel][sample], NOTE: raw_data_eeg is RAW DATA because we have CUSTOM FILTERS for artifact rejection
######def artifact_rejection_accel(raw_data_array_accel, channel_list_accel, sampling_rate, filename_output):
######    # === ACCEL artifact reject: MOVING WINDOW GENERAL ARTIFACTS
######    artifact_win_accel, artifact_step_accel, artifact_thresh_accel = 10, 2, 5
######    mirror_bandpass_data_artifact_accel = mirror_bandpass_filterAPI(raw_data_array_accel, len(raw_data_array_accel), 8, 1, 15, sampling_rate, 2)
######    artifact_time_strict_accel, artifact_mask_strict_accel = artifact_detect_moving_window(mirror_bandpass_data_artifact_accel, artifact_win_accel, artifact_step_accel, artifact_thresh_accel, 'on')
##########    print('artifact_time_strict_accel: ' + str(artifact_time_strict_accel))
########    save_artifact_rejection(str(filename_output), channel_list_accel, artifact_time_strict_accel)





########
############### input EEG: data[channel][sample], event_filter_value inputs are filter label below
############### filter label options: 'off', 'broadband', 'alpha'
##############def toggle_filter_eeg(data_eeg, event_filter_value, sampling_rate):
##############    data_eeg_output = np.zeros([len(data_eeg), len(data_eeg[0])])
##############    if event_filter_value == 'off':
##############        data_eeg_output = data_eeg
##############    if event_filter_value == 'broadband':
##############        for channel in range(0, len(data_eeg)):
##############            data_eeg_output[channel] = butter_bandpass_filterAPI(data_eeg[channel], .1, 30, sampling_rate, 4)
##############    if event_filter_value == 'delta/theta':
##############        for channel in range(0, len(data_eeg)):
##############            data_eeg_output[channel] = butter_bandpass_filterAPI(data_eeg[channel], .1, 10, sampling_rate, 8)
##############    if event_filter_value == 'alpha':
##############        for channel in range(0, len(data_eeg)):
##############            data_eeg_output[channel] = butter_bandpass_filterAPI(data_eeg[channel], 6, 15, sampling_rate, 8)
##############    if event_filter_value == 'beta':
##############        for channel in range(0, len(data_eeg)):
##############            data_eeg_output[channel] = butter_bandpass_filterAPI(data_eeg[channel], 15, 30, sampling_rate, 8)
##############
##############    return data_eeg_output
##############
############### input ACCEL: data[channel][sample], event_filter_value inputs are filter label below
############### filter label options: 'off', 'broadband', 'alpha'
##############def toggle_filter_accel(data_accel, event_filter_value, sampling_rate):
##############    data_accel_output = np.zeros([len(data_accel), len(data_accel[0])])
##############    if event_filter_value == 'off':
##############        data_accel_output = data_accel
##############    if event_filter_value == 'broadband':
##############        for channel in range(0, len(data_accel)):
##############            data_accel_output[channel] = butter_bandpass_filterAPI(data_accel[channel], 3, 15, sampling_rate, 4)
##############    if event_filter_value == 'delta/theta':
##############        for channel in range(0, len(data_accel)):
##############            data_accel_output[channel] = butter_bandpass_filterAPI(data_accel[channel], 1, 5, sampling_rate, 8)
##############    if event_filter_value == 'alpha':
##############        for channel in range(0, len(data_accel)):
##############            data_accel_output[channel] = butter_bandpass_filterAPI(data_accel[channel], 3, 10, sampling_rate, 8)
##############    if event_filter_value == 'beta':
##############        for channel in range(0, len(data_accel)):
##############            data_accel_output[channel] = butter_bandpass_filterAPI(data_accel[channel], 10, 15, sampling_rate, 8)
##############
##############    return data_accel_output
########
########
############# input ACCEL: data[channel][sample], event_filter_value inputs are filter label below
############# filter label options: 'off', 'broadband', 'alpha'
############def toggle_filter_accel(data_accel, lsl_accel, event_filter_value, sampling_rate):
############    data_accel_output = np.zeros([len(data_accel), len(lsl_accel)])
############    filter_range_accel = np.array([[3, 15], [1, 5], [3, 19], [5, 15]])
############    if event_filter_value == 'off':
############        data_output_accel = data_accel
############    if event_filter_value == 'broadband':
############        data_accel_output = filter_data_mne(data_eeg, lsl_accel, 3, 15, sampling_rate)
############    if event_filter_value == 'low':
############        data_accel_output = filter_data_mne(data_eeg, lsl_accel, 1, 5, sampling_rate)
############    if event_filter_value == 'mid':
############        data_accel_output = filter_data_mne(data_eeg, lsl_accel, 3, 10, sampling_rate)
############    if event_filter_value == 'high':
############        data_accel_output = filter_data_mne(data_eeg, lsl_accel, 10, 15, sampling_rate)
############
############    return data_accel_output
########






# === TF SCIPY SPECTRAL POWER
# input: raw_data_eeg[channel][time], raw_lsl_eeg[time]
# output: tf_power_trim_interp[channel][frequency][time], tf_freq_trim[frequency], tf_time_lsl[time], tf_artifact_mask[channel][time]
# output: tf_time_lsl[time], tf_artifact_mask_clean[time] where len(time) == len(tf_time) and tf_time_lsl is converted to lsl times and tf_artifact_mask has 0/1 for interpolated datapoints
def get_spectral_interp_2022_TESTING(raw_data_eeg, raw_lsl_eeg, artifactTime, band_range, sampling_rate, nfft, nperseg, interp_type, interpolation_toggle):
####    interpolation_toggle = 'on'     # options: on, off
##    interp_type = 'off'        # options: 'median', 'mean', 'mask'
    # === SPECT: scipy.signal.spectrogram (scaling='density' default, or 'spectrum'), input[channel][time], output[channel][freq][time]
    total_freq = [np.min(band_range), np.max(band_range)]
    tf_freq, tf_time, tf_power = scipy.signal.spectrogram(raw_data_eeg, fs=sampling_rate, nperseg=nperseg, nfft=nfft) # scaling='density' is default, or 'spectrum'
##    tf_power = 10*np.log10(tf_power)        # === optional scale to DB using 10*np.log10(data), sometimes produces nan, may need to remove them
    tf_power_trim, tf_freq_trim = get_time_frequency_singleband(tf_power, tf_freq, total_freq)              # === trim tf_power, tf_freq to frequencies of interest to limit computing power
    # convert time to lsl for tf_power_trim
    tf_time_samples = tf_time * sampling_rate               # convert spect_time to samples
    tf_time_lsl = np.zeros(len(tf_time_samples))            # optional: convert spect_time to lsl
    for value in range(0, len(tf_time_samples)):
        tf_time_lsl[value] = raw_lsl_eeg[int(tf_time_samples[value])]

    # ====== Interpolation basic variables
    if interpolation_toggle == 'on':
        # get tf_artifact_mask for interp vars, interpolate ACROSS CHANNEL tf_power_trim[channel][frequency][time] using tf_artifact_mask[channel][time]
        tf_artifact_mask = get_artifact_mask(artifactTime, tf_time_samples)
        tf_artifact_mask = connect_artifact_mask(tf_artifact_mask, 1)                # 1 = connect artifact times +/- 1
        tf_artifact_mask = reject_channel_artifact_mask(tf_artifact_mask, 0.5)       # 0.5 = 50% percent good data
        # interpolate and overwrite output variables
        tf_power_trim_interp = interpolate_tf_power_singleband_across_channel(tf_power_trim, tf_artifact_mask, str(interp_type))

    # ====== Bandpower tree
    # ====== Raw tree
    if interpolation_toggle == 'off':
        # === TF RAW: bandpower OVERTIME from tf_power_trim
        tf_bandpower_trim_raw_overtime = np.zeros([len(band_range), len(raw_data_eeg), len(tf_time)])
        for band in range(0, len(band_range)):
            tf_bandpower_trim_raw_overtime_buffer = get_time_frequency_overtime(tf_power_trim, tf_freq_trim, band_range[band])
            for channel in range(0, len(raw_data_eeg)):
                tf_bandpower_trim_raw_overtime[band][channel] = tf_bandpower_trim_raw_overtime_buffer[channel]
        # === TF RAW: bandpower AVETIME from tf_power_trim
        tf_bandpower_raw_avetime = np.zeros([len(band_range), len(raw_data_eeg)])
        for band in range(0, len(band_range)):
            for channel in range(0, len(raw_data_eeg)):
                tf_bandpower_raw_avetime[band][channel] = np.mean(tf_bandpower_trim_raw_overtime[band][channel])
    # ====== Interpolation tree
    if interpolation_toggle == 'on':
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
        # === TF INTERP: bandpower OVERTIME from tf_power_trim
        tf_bandpower_interp_avetime_clean = np.zeros([len(band_range), len(raw_data_eeg)])
        for band in range(0, len(band_range)):
            for channel in range(0, len(raw_data_eeg)):
                tf_bandpower_interp_avetime_clean[band][channel] = np.mean(tf_bandpower_interp_overtime_clean[band][channel])

##    # output
    if interpolation_toggle == 'off':
        return tf_bandpower_raw_avetime, tf_bandpower_trim_raw_overtime, tf_time_lsl
    if interpolation_toggle == 'on':
        return tf_bandpower_interp_avetime_clean, tf_bandpower_interp_overtime_clean, tf_time_lsl, tf_artifact_mask_clean










    


