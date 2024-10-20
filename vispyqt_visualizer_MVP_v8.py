# -*- coding: utf-8 -*-
# vispy: gallery 10:120:10
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
#
# =======================================
# Petal VisPyQt Visualizer V1 JG 2/5/23
# Three Class: CanvasWrapper (vispy), MyMainWindow (pyqt), DataSource (pyqt)
# Information Pathway: all information is one-way to avoid simultaneous updating from two sources
#   Pathway 1: MyMainWindow (pyqt GUI) --> DataSource (data processing) --> CanvasWrapper (vispy display)
#   Pathway 2: DataSource --> MyMainWindow required for things like focus/bpm updating straight from data --> GUI
#
# TROUBLESHOOTING:
# Call .emitters to list all emitters in this group: self.view[0].camera._viewbox.events.emitters
# =======================================
#
# NOTE: ALL MUSE HAVE IR, IR, RED FOR PPG SENSORS AFTER PLASTIC MUSE (E.G., MUSE S+)
# https://choosemuse.force.com/s/article/Muse-Products-Technical-Specifications?language=en_US

"""
Update data using timer events in a background thread
=====================================================

Update VisPy visualizations from timer events in a background QThread.

"""
import time
import sys
import math
import scipy
import numpy as np
from math import sin, pi
# pyqt
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
# vispy
from vispy import scene
from vispy.app import use_app
from vispy.app import MouseEvent
from vispy.geometry import Rect
from vispy.scene import SceneCanvas, visuals
from vispy.scene.cameras import PanZoomCamera, TurntableCamera, BaseCamera
# other
from vispyqt_visualizer_helperFuncs_v25 import *
from mne.filter import create_filter
from scipy.signal import lfilter, lfilter_zi
from pylsl import StreamInlet, resolve_byprop, local_clock

from mne.time_frequency import tfr_array_morlet
from vispyqt_visualizer_helperFuncsTest import *

import sip
import FreeSimpleGUI as sg
from pprint import pprint

import warnings

# EXAMPLE ONE WAY CODE
##    # CONNECT from MyMainWindow pyqt to Canvas, requires: win.new_data_filt.connect(canvas_wrapper.update_toggles)
##    def update_toggles(self, new_data_dict_filt):
##        if str(new_data_dict_filt["view_toggle"]) != str(self.toggle_view):
##            print('current_view: ' + str(self.toggle_view))
##            print('new_view: ' + str(new_data_dict_filt["view_toggle"]))
##        self.toggle_view = new_data_dict_filt["view_toggle"]

# print full numpy
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

# suppress MNE verbose to WARNING
mne.set_log_level('WARNING')

# https://www.codegrepper.com/code-examples/python/find+the+closest+value+in+an+array+python
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


### === Time-frequency variables notes
### nfft = frequency resolution in Hz, number of values in frequency array: len(f)=(nfft/2), f[0]=0, f[len(f)-1]=(samplingRate/2)
### nperseg = temporal resolution in time, number of times fft is performed per segment: greater nperseg = decreased len(t)
### Sxx[f][t] = Sxx[freq][sample], decreased nperseg to increase temporal resolution, increase nfft to increase frequency resolution
####nfft = samplingRate                                            # number of times within each segment the FFT is performed, default = 256
####nperseg = int(totalSamples/8)                                  # number of time segments, if totalSamples=256*4, nfft=256, nperseg=128, if totalSamples=256*2, nfft=256, nperseg=64
##nfft = 256                  # 256 here gives good frequency range, e.g. [8, 13] while at 128 frequency range is only [4, 6]
##nperseg = 64                # 128 is good too for less temporal resolution


# === PYSIMPLEGUI ===
layout = [
    [sg.Text("Welcome to Petal_Visualizer", size=(0, 1), key='Output_Data')],
    [sg.Button(' Play '), sg.Button(' Help '), sg.Button(' About '), sg.Button(' Exit ')],
    [sg.Text("", key='Warning')]]
window = sg.Window('Petal_Visualizer').Layout(layout)
# Event Loop
while True:
    event, values = window.Read()
    if event in (None, ' Exit '):
        exit()
        break
    elif event == ' Help ':
        layout_help_general = [
            [sg.Text('Description: Visualizer to display Muse data recorded by the Petal_GUI.')],
            [sg.Text("To use: ", size=(0, 1), key='Output_Data')],
            [sg.Text("1. Open Petal_GUI", size=(0, 1), key='Output_Data')],
            [sg.Text("2. Turn on Muse", size=(0, 1), key='Output_Data')],
            [sg.Text("3. Click Connect", size=(0, 1), key='Output_Data')],
            [sg.Text("4. When connected, click Play", size=(0, 1), key='Output_Data')],
            [sg.Button(' OK ')]]
        window_help_general = sg.Window('Petal Helper GUI V1 11_25_22', layout_help_general)
        while True:
            event_help_general, values_help_general = window_help_general.read()
            if event_help_general == sg.WIN_CLOSED or event_help_general == " OK ":
                window_help_general.close()
                break
    elif event == ' About ':
        layout_help_general = [
            [sg.Text('Description: Visualizer to display Muse data recorded by the Petal_GUI.')],
            [sg.Text('Author: James Glazer, Contact Info: glazerja1@gmail.com.')],
            [sg.Text('Please contact James Glazer for bugs, help, and/or questions. Thank you!')],
##            [sg.Text('Note: if data are longer than 1-2+ hours, the program may crash depending on your computer power.')],
            [sg.Button(' OK ')]]
        window_help_general = sg.Window('Petal Helper GUI V1 11_25_22', layout_help_general)
        while True:
            event_help_general, values_help_general = window_help_general.read()
            if event_help_general == sg.WIN_CLOSED or event_help_general == " OK ":
                window_help_general.close()
                break
    elif event == ' Play ':
        try:
            window["Warning"].update(value = "")
            print("looking for EEG stream...")
            streams_eeg = resolve_byprop('type', 'EEG', timeout=2)
            inlet_eeg = StreamInlet(streams_eeg[0], max_chunklen=12)
            break
        except:
            print("Could not find LSL stream.")
            window["Warning"].update(value = 'LSL stream not found, click Play to try again or Help for more information')
window.close()
# === PYSIMPLEGUI ===


# === LSL Variables
# EEG RAW: PetalMetrics -> Raw EEG
print("looking for EEG stream...")
streams_eeg = resolve_byprop('type', 'EEG', timeout=2)
##inlet_eeg = StreamInlet(streams_eeg[0], max_chunklen=24)
inlet_eeg = StreamInlet(streams_eeg[0], max_chunklen=12)
inlet_eeg_tf = StreamInlet(streams_eeg[0], max_chunklen=24)
# ACCEL RAW: PetalMetrics -> Raw ACCEL
print("looking for ACCEL stream...")
streams_accel = resolve_byprop('type', 'ACCEL', timeout=2)
inlet_accel = StreamInlet(streams_accel[0], max_chunklen=4)
# GYRO RAW: PetalMetrics -> Raw GYRO
print("looking for GYRO stream...")
streams_gyro = resolve_byprop('type', 'GYRO', timeout=2)
inlet_gyro = StreamInlet(streams_gyro[0], max_chunklen=4)
# PPG RAW: PetalMetrics -> Raw PPG
print("looking for PPG stream...")
streams_ppg = resolve_byprop('type', 'PPG', timeout=2)
inlet_ppg = StreamInlet(streams_ppg[0], max_chunklen=4)


# Define Device Variables
sampling_rate_eeg, sampling_rate_accel, sampling_rate_gyro, sampling_rate_ppg = 256, 52, 52, 64
##sampling_rate_eeg, sampling_rate_accel, sampling_rate_gyro, sampling_rate_ppg = 256, 54, 54, 64
ch_names_eeg, ch_names_accel, ch_names_gyro, ch_names_ppg = ['TP7', 'AF3', 'AF4', 'TP8'], ['X', 'Y', 'Z'], ['X', 'Y', 'Z'], ['am ', 'ir ', 'red']
n_chan_eeg, n_chan_accel, n_chan_gyro, n_chan_ppg = len(ch_names_eeg), len(ch_names_accel), len(ch_names_gyro), len(ch_names_ppg)

# window vars
window_seconds, cam_window_seconds = 30, 15
window_eeg, window_accel, window_gyro, window_ppg = int(sampling_rate_eeg * window_seconds), int(sampling_rate_accel * window_seconds), int(sampling_rate_gyro * window_seconds), int(sampling_rate_ppg * window_seconds)
cam_window_eeg, cam_window_accel, cam_window_gyro, cam_window_ppg = int(sampling_rate_eeg * cam_window_seconds), int(sampling_rate_accel * cam_window_seconds), int(sampling_rate_gyro * cam_window_seconds), int(sampling_rate_ppg * cam_window_seconds)

# EEG FFT/TF VARS
n_chan_eeg_fft = 4
nfft_eeg, nperseg_eeg = 256, 64
window_eeg_fft = 128
##cam_window_eeg_fft = 128
window_eeg_fft_trim = 256
n_chan_eeg_tf = 4
window_eeg_tf = int(window_eeg / 12)
##cam_window_eeg_tf = int(window_eeg / 12)
##window_eeg_tf_trim = int(sampling_rate_eeg * 3)

##nfft_eeg, nperseg_eeg = int(sampling_rate_eeg * 2), 64
##nfft_eeg, nperseg_eeg = int(sampling_rate_eeg * 2), 256
##window_eeg_tf_trim = int(sampling_rate_eeg * 6)

# TF ACCEL/GYRO/PPG
n_chan_accel_tf, n_chan_gyro_tf, n_chan_ppg_tf = 3, 3, 3
window_accel_tf = int(window_accel / 4)
window_accel_tf_trim = int(sampling_rate_accel * 3)
window_gyro_tf = int(window_gyro / 4)
window_gyro_tf_trim = int(sampling_rate_gyro * 3)
window_ppg_tf = int(window_ppg / 4)
window_ppg_tf_trim = int(sampling_rate_ppg * 3)
##cam_window_accel_tf = int(window_ppg / 4)
##cam_window_gyro_tf = int(window_ppg / 4)
##cam_window_ppg_tf = int(window_ppg / 4)

# FFT ACCEL/GYRO
n_chan_accel_fft, n_chan_gyro_fft, n_chan_ppg_fft = 3, 3, 3 
window_accel_fft = 26
window_accel_fft_trim = 52
window_gyro_fft = 26
window_gyro_fft_trim = 52
##cam_window_accel_fft = 26
##cam_window_gyro_fft = 26
nfft_accel, nperseg_accel = 52, 26
nfft_gyro, nperseg_gyro = 52, 26
nfft_ppg, nperseg_ppg = 64, 32

# FFT PPG
window_ppg_fft = int((sampling_rate_ppg * 10) / 2)
window_ppg_fft_trim = int(sampling_rate_ppg * 10)
##cam_window_ppg_fft = int((sampling_rate_ppg * 10) / 2)

# === EEG PSD
freqLow_psd_eeg, freqHigh_psd_eeg, freqNum_psd_eeg = 1, 50, 50
freqs_psd_eeg = np.arange(50) + 1.0
##freqs_psd_eeg = np.logspace(*np.log10([freqLow_psd_eeg, freqHigh_psd_eeg]), num=freqNum_psd_eeg)

# Define Canvas Variables
CANVAS_SIZE = (800, 600)  # (width, height)
##LINE_COLOR_CHOICES = ["black", "red", "blue", "green", "purple"]
LINE_COLOR_CHOICES = ["black", "red", "blue", "green", "purple"]
IMAGE_SHAPE = (freqNum_psd_eeg, window_eeg)  # (height, width)
COLORMAP_CHOICES = ["jet", "reds", "blues"]

# === ACCEL/GYRO/PPG PSD
freqs_psd_accel = np.arange(26) + 1.0
freqLow_psd_accel, freqHigh_psd_accel, freqNum_psd_accel = 1, 26, 26
##freqs_psd_accel = np.logspace(*np.log10([freqLow_psd_accel, freqHigh_psd_accel]), num=freqNum_psd_accel)
IMAGE_SHAPE_accel = (freqNum_psd_accel, window_accel)  # (height, width)
freqs_psd_gyro = np.arange(26) + 1.0
freqLow_psd_gyro, freqHigh_psd_gyro, freqNum_psd_gyro = 1, 26, 26
##freqs_psd_gyro = np.logspace(*np.log10([freqLow_psd_gyro, freqHigh_psd_gyro]), num=freqNum_psd_gyro)
IMAGE_SHAPE_gyro = (freqNum_psd_gyro, window_gyro)  # (height, width)

##freqs_psd_ppg = np.arange(31) + 1.0
####freqLow_psd_ppg, freqHigh_psd_ppg, freqNum_psd_ppg = .5, 3, 60
##freqLow_psd_ppg, freqHigh_psd_ppg, freqNum_psd_ppg = .6, 2.5, 120


##freqLow_psd_ppg, freqHigh_psd_ppg, freqNum_psd_ppg = .5, 3, 240
freqLow_psd_ppg, freqHigh_psd_ppg, freqNum_psd_ppg = .5, 3, 180


##freqLow_psd_ppg, freqHigh_psd_ppg, freqNum_psd_ppg = 1, 31, 31
freqs_psd_ppg = np.logspace(*np.log10([freqLow_psd_ppg, freqHigh_psd_ppg]), num=freqNum_psd_ppg)
IMAGE_SHAPE_ppg = (freqNum_psd_ppg, window_ppg)  # (height, width)

# ================== FINAL FILTERS ACCEL/GYRO/PPG
eeg_filter_list_final = np.array([[3, 40], [1, 8], [6, 15], [15, 30], [25, 50]])
accel_filter_list_final = np.array([[1, 25], [1, 5], [3, 8], [6, 15], [12, 25]])
gyro_filter_list_final = np.array([[1, 25], [1, 5], [3, 8], [6, 15], [12, 25]])
##ppg_filter_list_final = np.array([[2, 30], [1, 5], [3, 8], [6, 15], [12, 30]])
ppg_filter_list_final = np.array([[.5, 30], [.5, 3], [.5, 5], [.5, 10], [12, 30]])
eeg_filter_list_text_final = ['off', 'default', 'delta/theta', 'alpha', 'beta', 'gamma']
accel_filter_list_text_final = ['off_accel', 'default_accel', 'low_accel', 'mid_accel', 'high_accel', 'highest_accel']
gyro_filter_list_text_final = ['off_gyro', 'default_gyro', 'low_gyro', 'mid_gyro', 'high_gyro', 'highest_gyro']
ppg_filter_list_text_final = ['off_ppg', 'default_ppg', 'low_ppg', 'mid_ppg', 'high_ppg', 'highest_ppg']
# === Define Band Ranges
eeg_band_range_text = ['delta', 'theta', 'alpha', 'beta', 'gamma']
##accel_filter_list_text_final = ['off_accel', 'default_accel', 'low_accel', 'mid_accel', 'high_accel', 'highest_accel']
##gyro_filter_list_text_final = ['off_gyro', 'default_gyro', 'low_gyro', 'mid_gyro', 'high_gyro', 'highest_gyro']
##ppg_filter_list_text_final = ['off_ppg', 'default_ppg', 'low_ppg', 'mid_ppg', 'high_ppg', 'highest_ppg']
accel_band_list_text_final = ['default_accel', 'low_accel', 'mid_accel', 'high_accel', 'highest_accel']
gyro_band_list_text_final = ['default_gyro', 'low_gyro', 'mid_gyro', 'high_gyro', 'highest_gyro']
ppg_band_list_text_final = ['default_ppg', 'low_ppg', 'mid_ppg', 'high_ppg', 'highest_ppg']

eeg_band_list_final = np.array([[1, 3], [3, 8], [8, 13], [13, 30], [30, 50]])
accel_band_list_final = np.array([[1, 25], [1, 5], [3, 8], [6, 15], [12, 25]])
gyro_band_list_final = np.array([[1, 25], [1, 5], [3, 8], [6, 15], [12, 25]])
##ppg_band_list_final = np.array([[2, 30], [1, 5], [3, 8], [6, 15], [12, 30]])
ppg_band_list_final = np.array([[.5, 30], [.5, 3], [.5, 5], [.5, 10], [12, 30]])

######eeg_band_list_final = np.array([[1, 3], [3, 8], [8, 13], [13, 30], [30, 50]])
######accel_band_list_final = np.array([[1, 25], [1, 5], [3, 8], [6, 15], [12, 25]])
######gyro_band_list_final = np.array([[1, 25], [1, 5], [3, 8], [6, 15], [12, 25]])
########ppg_band_list_final = np.array([[2, 30], [1, 5], [3, 8], [6, 15], [12, 30]])
######ppg_band_list_final = np.array([[.5, 30], [.5, 3], [.5, 5], [.5, 10], [12, 30]])

class CanvasWrapper():

    def __init__(self):

        # define canvas
        self.canvas = SceneCanvas(size=CANVAS_SIZE)
        self.grid = self.canvas.central_widget.add_grid()

        # define vars

        self.cam_toggle_count = 0
        self.update_x_axis = False
        
        self.play_bool, self.corner_widget_bool_canvas = True, False
        self.cam_toggle, self.toggle_device_view = "Auto", "EEG_Time_Series"
        self.last_cam_toggle_canvas, self.current_cam_toggle_canvas = "Auto", "Auto"
        self.current_artifact_filt, self.current_artifact_filt_accel, self.current_artifact_filt_gyro = 0, 0, 0
##        self.play_bool = True
##        self.corner_widget_bool_canvas = False
##        self.cam_toggle = "Auto"
##        self.toggle_device_view = "EEG_Time_Series"
##        self.last_cam_toggle_canvas = "Auto"
##        self.current_cam_toggle_canvas = "Auto"
##        self.current_artifact_filt = 0
##        self.current_artifact_filt_accel = 0
##        self.current_artifact_filt_gyro = 0

        self.last_time_test = local_clock()
        self.line_bool_canvas = [True]*n_chan_eeg

        # ========= COLOR VIEW CAMERA UPDATED NEW =========
        self.view, self.line, self.line_color, self.y_axis_all = [], [], [], []
        line_data_buffer = np.swapaxes(np.asarray([np.arange(window_eeg), np.zeros(window_eeg)]), 0, 1)
        color_list = [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 0.0, 1.0]]             # color_list = [black, red, blue, green]  
        for channel in range(0, n_chan_eeg):
            color = np.ones((window_eeg, 4), dtype=np.float32)
            color[:, 0:4] = color_list[channel]
##            color[:, 0] = np.linspace(0, 1, window_eeg)
##            color[:, 1] = color[::-1, 0]
            self.line_color.append(color)
            self.view.append(self.grid.add_view(channel, 0, bgcolor='#c0c0c0'))
            self.line.append(visuals.Line(line_data_buffer, parent=self.view[channel].scene, color=color)) # color=LINE_COLOR_CHOICES[channel]))
            self.view[channel].camera = scene.PanZoomCamera(rect=(0, -25, window_eeg, 50), interactive=True)
            self.view[channel].camera.rect.left, self.view[channel].camera.rect.right = cam_window_eeg, window_eeg
            # === DISCONNECT MOUSE EVENTS from auto control from vispy so you can manually edit below
            self.view[channel].camera._viewbox.events.mouse_move.disconnect(self.view[channel].camera.viewbox_mouse_event)
            self.view[channel].camera._viewbox.events.mouse_press.disconnect(self.view[channel].camera.viewbox_mouse_event)
            self.view[channel].camera._viewbox.events.mouse_wheel.disconnect(self.view[channel].camera.viewbox_mouse_event)
            # === SELF.Y_AXIS VIEW BASIC
##            self.y_axis_all.append(scene.AxisWidget(orientation='right', axis_label='uV', axis_label_margin=-1))
            self.y_axis_all.append(scene.AxisWidget(orientation='right', axis_label='uV', axis_label_margin=-1, axis_color='black', text_color='black'))
            self.view[channel].add_widget(self.y_axis_all[channel])
            self.y_axis_all[channel].link_view(self.view[channel])

        # CONNECT MOUSE EVENTS to viewbox_mouse_event for manual scaling with mouse: self.canvas.events = mouse_wheel, mouse_press, mouse_move
        self.canvas.events.mouse_press.connect(self.viewbox_mouse_event)
        self.canvas.events.mouse_move.connect(self.viewbox_mouse_event)
        self.canvas.events.mouse_wheel.connect(self.viewbox_mouse_event)

        # ========= COLOR VIEW CAMERA UPDATED NEW =========
        # === SELF.X_AXIS VIEW
        self.view_x_axis = self.grid.add_view(len(self.view), 0, bgcolor='black')
        self.view_x_axis.camera = "panzoom"
        self.view_x_axis.camera.set_range(x=(int(window_eeg / sampling_rate_eeg), 0), y=(-1, 1), margin=0)
##        self.view_x_axis.camera.set_range(x=(int(cam_window_eeg / sampling_rate_eeg), 0), y=(-1, 1), margin=0)
        # SELF.X_AXIS WIDGET
        self.x_axis = scene.AxisWidget(orientation='bottom')
        domain_list = list(self.x_axis.axis._domain)
        domain_list[0], domain_list[1] = int(window_eeg / sampling_rate_eeg), 0
##        domain_list[0], domain_list[1] = int(cam_window_eeg / sampling_rate_eeg), 0
        self.x_axis.axis._domain = tuple(domain_list)
        self.view_x_axis.camera.rect.left, self.view_x_axis.camera.rect.right = int(cam_window_eeg / sampling_rate_eeg), 0
        self.x_axis.stretch = (1, 0.2)
        self.grid.add_widget(self.x_axis, row=len(self.view), col=0)
        # === VIEW SELF.X_AXIX disconnect/connect mouse events
        self.view_x_axis.camera._viewbox.events.mouse_move.disconnect(self.view_x_axis.camera.viewbox_mouse_event)
        self.view_x_axis.camera._viewbox.events.mouse_press.disconnect(self.view_x_axis.camera.viewbox_mouse_event)
        self.view_x_axis.camera._viewbox.events.mouse_wheel.disconnect(self.view_x_axis.camera.viewbox_mouse_event)
        # define limits
        rect_buffer_x_axis_original = Rect(self.view_x_axis.camera.rect)
##        self.view_x_axis_left_limit = rect_buffer_x_axis_original.left
        self.view_x_axis_left_limit, self.view_x_axis_right_limit = int(rect_buffer_x_axis_original.left * 2), rect_buffer_x_axis_original.right





        # ========= ARTIFACTS/COLOR NEW =========
        # COLOR define filter FULL vars
        self.eeg_filter_list_range = eeg_filter_list_final
        self.marker_line_values, self.current_artifact_pos = [[], [], [], []], [[], [], [], []]
        self.data_eeg_filter_all, self.lsl_eeg_corrected, self.filt_time_index_corrected, self.bf_eeg_all, self.zi_eeg_all, self.filt_state_eeg_all = create_data_filter_variables(
            window_eeg, n_chan_eeg, sampling_rate_eeg, self.eeg_filter_list_range)
        # === define self.bf_eeg_index_offset to send for artifact reject bf offset time lag correction, use len(self.eeg_filter_list_range) + 1 because we add 0 = raw
        self.bf_eeg_index_offset = np.zeros(len(self.eeg_filter_list_range) + 1)
        for filt in range(0, len(self.eeg_filter_list_range)):
            filt_time_buffer = int(len(self.bf_eeg_all[filt]) / 2)# * (1 / sampling_rate_eeg)
            self.bf_eeg_index_offset[filt + 1] = filt_time_buffer
        # define eeg window line color arrays
        self.line_color_plot = []
        for filt in range(0, len(self.eeg_filter_list_range) + 1):
            self.line_color_plot.append(self.line_color)
        # COLOR: LONG FULL line COLOR data where len() = window_eeg + max(len(bf_all))
        self.line_color_long = []
        self.window_eeg_long = int(window_eeg + np.max(self.bf_eeg_index_offset))
        for channel in range(0, n_chan_eeg):
            color = np.ones((self.window_eeg_long, 4), dtype=np.float32)
            color[:, 0:4] = color_list[channel]
            self.line_color_long.append(color)
        # update range
        self.line_color_range = []
        for filt in range(0, len(self.bf_eeg_index_offset)):
            range_buffer = [int(self.window_eeg_long - window_eeg - self.bf_eeg_index_offset[filt]), int(self.window_eeg_long - self.bf_eeg_index_offset[filt])]
            self.line_color_range.append(range_buffer)
        self.line_color_range = np.array(self.line_color_range)
        # ========= ARTIFACTS/COLOR NEW =========

        # ========= ACCEL ARTIFACTS/COLOR NEW =========
        # COLOR define filter FULL vars
        self.accel_filter_list_range = accel_filter_list_final
        self.marker_line_values_accel, self.current_artifact_pos_accel = [[], [], []], [[], [], []]
        self.data_accel_filter_all, self.lsl_accel_corrected, self.filt_time_index_accel_corrected, self.bf_accel_all, self.zi_accel_all, self.filt_state_accel_all = create_data_filter_variables(
            window_accel, n_chan_accel, sampling_rate_accel, self.accel_filter_list_range)
        # === define self.bf_accel_index_offset to send for artifact reject bf offset time lag correction, use len(self.accel_filter_list_range) + 1 because we add 0 = raw
        self.bf_accel_index_offset = np.zeros(len(self.accel_filter_list_range) + 1)
        for filt in range(0, len(self.accel_filter_list_range)):
            filt_time_buffer = int(len(self.bf_accel_all[filt]) / 2)# * (1 / sampling_rate_accel)
            self.bf_accel_index_offset[filt + 1] = filt_time_buffer
        # DEFINE COLOR ACCEL
        self.line_color_accel = []
        color_list = [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 0.0, 1.0]]             # color_list = [black, red, blue, green]  
        for channel in range(0, n_chan_accel):
            color = np.ones((window_accel, 4), dtype=np.float32)
            color[:, 0:4] = color_list[channel]
            self.line_color_accel.append(color)
        # define eeg window line color arrays
        self.line_color_plot_accel = []
        for filt in range(0, len(self.accel_filter_list_range) + 1):
            self.line_color_plot_accel.append(self.line_color_accel)
        self.line_color_long_accel = []
        self.window_accel_long = int(window_accel + np.max(self.bf_accel_index_offset))
        for channel in range(0, n_chan_accel):
            color = np.ones((self.window_eeg_long, 4), dtype=np.float32)
            color[:, 0:4] = color_list[channel]
            self.line_color_long_accel.append(color)
        # update range
        self.line_color_range_accel = []
        for filt in range(0, len(self.bf_accel_index_offset)):
            range_buffer = [int(self.window_accel_long - window_accel - self.bf_accel_index_offset[filt]), int(self.window_accel_long - self.bf_accel_index_offset[filt])]
            self.line_color_range_accel.append(range_buffer)
        self.line_color_range_accel = np.array(self.line_color_range_accel)
        # ========= ACCEL ARTIFACTS/COLOR NEW =========

        # ========= GYRO ARTIFACTS/COLOR NEW =========
        # COLOR define filter FULL vars
        self.gyro_filter_list_range = gyro_filter_list_final
        self.marker_line_values_gyro, self.current_artifact_pos_gyro = [[], [], []], [[], [], []]
        self.data_gyro_filter_all, self.lsl_gyro_corrected, self.filt_time_index_gyro_corrected, self.bf_gyro_all, self.zi_gyro_all, self.filt_state_gyro_all = create_data_filter_variables(
            window_gyro, n_chan_gyro, sampling_rate_gyro, self.gyro_filter_list_range)
        # === define self.bf_gyro_index_offset to send for artifact reject bf offset time lag correction, use len(self.gyro_filter_list_range) + 1 because we add 0 = raw
        self.bf_gyro_index_offset = np.zeros(len(self.gyro_filter_list_range) + 1)
        for filt in range(0, len(self.gyro_filter_list_range)):
            filt_time_buffer = int(len(self.bf_gyro_all[filt]) / 2)# * (1 / sampling_rate_gyro)
            self.bf_gyro_index_offset[filt + 1] = filt_time_buffer
        # DEFINE COLOR gyro
        self.line_color_gyro = []
        color_list = [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 0.0, 1.0]]             # color_list = [black, red, blue, green]  
        for channel in range(0, n_chan_gyro):
            color = np.ones((window_gyro, 4), dtype=np.float32)
            color[:, 0:4] = color_list[channel]
            self.line_color_gyro.append(color)
        # define eeg window line color arrays
        self.line_color_plot_gyro = []
        for filt in range(0, len(self.gyro_filter_list_range) + 1):
            self.line_color_plot_gyro.append(self.line_color_gyro)
        self.line_color_long_gyro = []
        self.window_gyro_long = int(window_gyro + np.max(self.bf_gyro_index_offset))
        for channel in range(0, n_chan_gyro):
            color = np.ones((self.window_eeg_long, 4), dtype=np.float32)
            color[:, 0:4] = color_list[channel]
            self.line_color_long_gyro.append(color)
        # update range
        self.line_color_range_gyro = []
        for filt in range(0, len(self.bf_gyro_index_offset)):
            range_buffer = [int(self.window_gyro_long - window_gyro - self.bf_gyro_index_offset[filt]), int(self.window_gyro_long - self.bf_gyro_index_offset[filt])]
            self.line_color_range_gyro.append(range_buffer)
        self.line_color_range_gyro = np.array(self.line_color_range_gyro)
        # ========= GYRO ARTIFACTS/COLOR NEW =========

##        for property, value in vars(self.x_axis).items():
##            print(property, ":", value)
##        print(correct)

    # === MULTICHAN MANUAL ZOOM taken from panzoom.py in vispy lib
    def viewbox_mouse_event(self, event):


        self.x_axis.link_view(self.view_x_axis)


        # ===== Camera Mouse Events Details
        # Bandpower/FFT share Mouse_Wheel, Bandpower/TimeSeries share Mouse_Move
        # Time_Series:
        #   Mouse_Wheel: scale top/bottom equally, left/right = fixed
        #   Mouse_Move: scale LEFT right = 0, top/bottom = fixed
        # FFT:
        #   Mouse_Wheel: scale top, bottom = 0, left/right = fixed
        #   Mouse_Move: scale RIGHT, left = 0, top/bottom = fixed
        # Bandpower:
        #   Mouse_Wheel: scale top, bottom = 0, left/right = fixed
        #   Mouse_Move: scale LEFT, right = 0, top/bottom = fixed

        # REFACTOR first get the device n_chan and n_window sampling_rate, add zoom factors, then paramaters for mouse camera control: zoom, rect_top_limit, rect_left_limit
        mouse_dict_list = {
                     "EEG_Time_Series":         [n_chan_eeg,      window_eeg,       sampling_rate_eeg,   int(window_eeg - sampling_rate_eeg)],
                     "ACCEL_Time_Series":       [n_chan_accel,    window_accel,     sampling_rate_accel, int(window_accel - sampling_rate_accel)],
                     "GYRO_Time_Series":        [n_chan_gyro,     window_gyro,      sampling_rate_gyro,  int(window_gyro - sampling_rate_gyro)],
                     "PPG_Time_Series":         [n_chan_ppg,      window_ppg,       sampling_rate_ppg,   int(window_ppg - sampling_rate_ppg)],
                     "EEG_FFT":                 [1,               window_eeg_fft,   sampling_rate_eeg,   int(window_eeg_fft - sampling_rate_eeg)],
                     "ACCEL_FFT":               [1,               window_accel_fft, sampling_rate_accel, int(window_accel_fft - sampling_rate_accel)],
                     "GYRO_FFT":                [1,               window_gyro_fft,  sampling_rate_gyro,  int(window_gyro_fft - sampling_rate_gyro)],
                     "PPG_FFT":                 [1,               window_ppg_fft,   sampling_rate_ppg,   int(window_ppg_fft - sampling_rate_ppg)],
                     "EEG_TF_BandChan":         [n_chan_eeg_tf,   window_eeg_tf,    sampling_rate_eeg,   int(window_eeg_tf - 32)],
                     "ACCEL_TF_BandChan":       [n_chan_accel_tf, window_accel_tf,  sampling_rate_accel, int(window_accel_tf - 32)],
                     "GYRO_TF_BandChan":        [n_chan_gyro_tf,  window_gyro_tf,   sampling_rate_gyro,  int(window_gyro_tf - 32)],
                     "PPG_TF_BandChan":         [n_chan_ppg_tf,   window_ppg_tf,    sampling_rate_ppg,   int(window_ppg_tf - 32)],
                     "EEG_TF_Band":             [1,               window_eeg_tf,    sampling_rate_eeg,   int(window_eeg_tf - 32)],
                     "ACCEL_TF_Band":           [1,               window_accel_tf,  sampling_rate_accel, int(window_accel_tf - 32)],
                     "GYRO_TF_Band":            [1,               window_gyro_tf,   sampling_rate_gyro,  int(window_gyro_tf - 32)],
                     "PPG_TF_Band":             [1,               window_ppg_tf,    sampling_rate_ppg,   int(window_ppg_tf - 32)],
                     "EEG_PSD":                 [n_chan_eeg,      window_eeg,       sampling_rate_eeg,   int(window_eeg - sampling_rate_eeg)], 
                     "ACCEL_PSD":               [n_chan_accel,    window_accel,     sampling_rate_accel, int(window_accel - sampling_rate_accel)], 
                     "GYRO_PSD":                [n_chan_gyro,     window_gyro,      sampling_rate_gyro,  int(window_gyro - sampling_rate_gyro)], 
                     "PPG_PSD":                 [n_chan_ppg,      window_ppg,       sampling_rate_ppg,   int(window_ppg - sampling_rate_ppg)]}

        n_chan_buffer = mouse_dict_list[str(self.toggle_device_view)][0]
        window_buffer = mouse_dict_list[str(self.toggle_device_view)][1]
        sampling_rate_buffer = mouse_dict_list[str(self.toggle_device_view)][2]
        rect_left_limit = mouse_dict_list[str(self.toggle_device_view)][3]

        # TF VAR
        if self.toggle_device_view in ["EEG_TF_Band", "EEG_TF_BandChan", "ACCEL_TF_Band", "ACCEL_TF_BandChan", "GYRO_TF_Band", "GYRO_TF_BandChan", "PPG_TF_Band", "PPG_TF_BandChan"]:
            rect_left_limit = int(window_buffer - 32)
        window_buffer = window_buffer - 1               # NOTE: window_buffer - 1 because we include 0

        # === PSD MOUSE EVENTS
        if self.toggle_device_view in ["EEG_PSD", "ACCEL_PSD", "GYRO_PSD", "PPG_PSD"]:
            # == Mouse_Wheel_Event: scale top/bottom equally, left/right = fixed
            if event.type == 'mouse_wheel':
                self.cam_toggle = "Unlock"
                zoom_in, zoom_out = 1 - (.05 * abs(event.delta[1])), 1 + (.05 * abs(event.delta[1]))
                # === VIEW TOP scale using self.psd_scale_toggle
##                print('zoom_in, zoom_out: ' + str(zoom_in) + ', ' + str(zoom_out))
                if event.delta[1] < 0:                  # zoom out
                    self.psd_scale = self.psd_scale * zoom_out
                if event.delta[1] > 0:                  # zoom in
                    self.psd_scale = self.psd_scale * zoom_in
                for channel in range(0, n_chan_eeg):
                    rect_buffer = Rect(self.view[channel].camera.rect)
                    self.view[channel].camera.rect = rect_buffer
##                print('self.psd_scale: ' + str(self.psd_scale))

            # === Mouse_Move_Event: scale LEFT right = 0, top/bottom = fixed
            if event.type == 'mouse_move':
                if event.press_event is None:
                    return
                if 1 in event.buttons:
                    # === PSD: X-AXIS view flip
                    rect_buffer_x = Rect(self.view_x_axis.camera.rect)
                    p1, p2 = np.array(event.last_event.pos)[:2], np.array(event.pos)[:2]
                    p1s, p2s = self.view_x_axis.camera._transform.imap(p1), self.view_x_axis.camera._transform.imap(p2)
                    self.view_x_axis.camera.pan(p1s - p2s)
                    rect_x = Rect(self.view_x_axis.camera.rect)
                    rect_x.right, rect_x.top, rect_x.bottom = 0, rect_buffer_x.top, rect_buffer_x.bottom
                    if rect_x.left > 30:     # stop pan in
                        rect_x.left = 30
                    self.view_x_axis.camera.rect = rect_x
                    # === PSD: CHANNEL Camera Pan
                    for channel in range(0, n_chan_buffer):
                        rect_buffer = Rect(self.view[channel].camera.rect)
                        p1, p2 = np.array(event.last_event.pos)[:2], np.array(event.pos)[:2]
                        p1s, p2s = self.view[channel].camera._transform.imap(p1), self.view[channel].camera._transform.imap(p2)
                        self.view[channel].camera.pan(p1s - p2s)
                        rect = Rect(self.view[channel].camera.rect)
                        rect.right, rect.top, rect.bottom = window_buffer, rect_buffer.top, rect_buffer.bottom
                        if rect.left < self.view_psd_left_limit:                         # stop pan out
                            rect.left = self.view_psd_left_limit
                        self.view[channel].camera.rect = rect
                    event.handled = True

                # if you want another zoom option with other mouse button
                elif 2 in event.buttons:
##                    print('Mouse_Button_Event_TWO_PSD: ' + str(event.delta))
                    p1c, p2c = np.array(event.last_event.pos)[:2], np.array(event.pos)[:2]
                    subtract_buffer = np.array(p1c - p2c)[1] * .05
                    for channel in range(0, n_chan_buffer):
                        rect_buffer = Rect(self.view[channel].camera.rect)
                        rect = Rect(self.view[channel].camera.rect)
                        rect.top = rect.top - subtract_buffer
                        if self.toggle_device_view == "EEG_PSD":
                            if rect.top > freqHigh_psd_eeg:
                                rect.top = freqHigh_psd_eeg
                            if rect.top < 3:
                                rect.top = 3
                        elif self.toggle_device_view == "ACCEL_PSD":
                            if rect.top > freqHigh_psd_accel:
                                rect.top = freqHigh_psd_accel
                            if rect.top < 3:
                                rect.top = 3
                        elif self.toggle_device_view == "GYRO_PSD":
                            if rect.top > freqHigh_psd_gyro:
                                rect.top = freqHigh_psd_gyro
                            if rect.top < 3:
                                rect.top = 3
                        if self.toggle_device_view == "PPG_PSD":
                            if rect.top > freqHigh_psd_ppg:
                                rect.top = freqHigh_psd_ppg
                            if rect.top < .5:
                                rect.top = .5
                        rect.bottom = 0
                        self.view[channel].camera.rect = rect
                    event.handled = True
                else:
                    event.handled = False

            # === Mouse_Press_Event
            elif event.type == 'mouse_press':
                event.handled = event.button in [1, 2]
            else:
                event.handled = False

        # === ALL DEVICE/VIEW MOUSE EVENTS
        else:
            # === MOUSE WHEEL EVENTS
            if event.type == 'mouse_wheel':
                self.cam_toggle = "Unlock"
                zoom_in, zoom_out = 1 - (.05 * abs(event.delta[1])), 1 + (.05 * abs(event.delta[1]))
                for channel in range(0, n_chan_buffer):
                    rect_buffer = Rect(self.view[channel].camera.rect)
                    rect = Rect(self.view[channel].camera.rect)
                    # == TIME_SERIES Mouse_Wheel_Event: scale top/bottom equally, left/right = fixed
                    if self.toggle_device_view in ["EEG_Time_Series", "ACCEL_Time_Series", "GYRO_Time_Series", "PPG_Time_Series"]:
                        if event.delta[1] < 0:                  # zoom out
                            rect.bottom, rect.top = rect.bottom * zoom_out, rect.top * zoom_out
                        if event.delta[1] > 0:                  # zoom in
                            rect.bottom, rect.top = rect.bottom * zoom_in, rect.top * zoom_in
                        rect.right = window_buffer
                        if self.toggle_device_view == "EEG_Time_Series":                        # EEG TIME SERIES
                            rect_top_limit = 10
                        else:                                                                   # ACCEL/GYRO/PPG TIME SERIES
                            rect_top_limit = .5
                        # control for negative height, REVERSES AT 0.5 FOR SOME REASON THOUGH REVERSES AT < 0.5, so we use 0.6 here
                        if rect.top < rect_top_limit:           # stop zoom in by reseting rect
                            rect.bottom, rect.top = rect_buffer.bottom, rect_buffer.top
                        self.view[channel].camera.rect = rect
                    # == FFT Mouse_Wheel_Event: scale top, bottom = 0, left/right = fixed
                    if self.toggle_device_view in ["EEG_FFT", "ACCEL_FFT", "GYRO_FFT", "PPG_FFT"]:
                        if event.delta[1] < 0:                  # zoom out
                            rect.top = rect.top * zoom_out
                        if event.delta[1] > 0:                  # zoom in
                            rect.top = rect.top * zoom_in
                        rect.left = 0       # FFT ONLY
                        # control for negative height by resetting rect
##                        print('rect.top: ' + str(rect.top))
                        if rect.top < 2:                # control for negative height by resetting rect
                            rect.top = rect_buffer.top
                        rect.bottom = 0
                        self.view[channel].camera.rect = rect
                        channel = n_chan_buffer
                    # === TF Mouse_Wheel_Event: scale top, bottom = 0, left/right = fixed
                    if self.toggle_device_view in ["EEG_TF_Band", "EEG_TF_BandChan", "ACCEL_TF_Band", "ACCEL_TF_BandChan", "GYRO_TF_Band", "GYRO_TF_BandChan", "PPG_TF_Band", "PPG_TF_BandChan"]:
                        if event.delta[1] < 0:                  # zoom out
                            rect.top = rect.top * zoom_out
                        if event.delta[1] > 0:                  # zoom in
                            rect.top = rect.top * zoom_in
                        rect.right = window_buffer
                        # control for negative height by resetting rect
                        if rect.top < 2:
                            rect.top = rect_buffer.top
                        self.view[channel].camera.rect = rect

            # === MOUSE MOVE EVENTS
            if event.type == 'mouse_move':
                if event.press_event is None:
                    return
                if 1 in event.buttons:
                    # === X-AXIS VIEW UPDATE
                    rect_buffer_x = Rect(self.view_x_axis.camera.rect)
                    p1, p2 = np.array(event.last_event.pos)[:2], np.array(event.pos)[:2]
                    p1s, p2s = self.view_x_axis.camera._transform.imap(p1), self.view_x_axis.camera._transform.imap(p2)
##                    print('p1, p2: ' + str(p1) + ', ' + str(p2))
##                    print('p1s, p2s: ' + str(p1s) + ', ' + str(p2s))
##                    print('p1s - p2s: ' + str(p1s - p2s))
##                    print('type: p1, p2, p1s, p2s: ' + str(type(p1)) + ', ' + str(type(p2)) + ', ' + str(type(p1s)) + ', ' + str(type(p2s)))
##                    print('p1.shape: ' + str(p1.shape))
                    self.view_x_axis.camera.pan(p1s - p2s)
                    rect_x = Rect(self.view_x_axis.camera.rect)
                    # === CHANNEL VIEW LINE/DATA UPDATE
                    for channel in range(0, n_chan_buffer):
                        rect_buffer = Rect(self.view[channel].camera.rect)
                        p1, p2 = np.array(event.last_event.pos)[:2], np.array(event.pos)[:2]
                        p1s, p2s = self.view[channel].camera._transform.imap(p1), self.view[channel].camera._transform.imap(p2)
                        self.view[channel].camera.pan(p1s - p2s)
                        rect = Rect(self.view[channel].camera.rect)
                        # === Time_Series Mouse_Move_Event: scale LEFT right = 0, top/bottom = fixed
                        if self.toggle_device_view in ["EEG_Time_Series", "ACCEL_Time_Series", "GYRO_Time_Series", "PPG_Time_Series"]:
                            rect_x.right, rect_x.top, rect_x.bottom = 0, rect_buffer_x.top, rect_buffer_x.bottom
                            if rect_x.left > 30:     # stop pan out
                                rect_x.left = 30
                            if rect_x.left < 1:     # stop pan in
                                rect_x.left = 1
                            self.view_x_axis.camera.rect = rect_x
                            rect.right, rect.top, rect.bottom = window_buffer, rect_buffer.top, rect_buffer.bottom
                            if rect.left > rect_left_limit:     # stop pan in
                                rect.left = rect_left_limit
                            if rect.left < 0:                   # stop pan out
                                rect.left = 0
                            self.view[channel].camera.rect = rect
                        # === TF Mouse_Move_Event: scale LEFT right = 0, top/bottom = fixed
                        if self.toggle_device_view in ["EEG_TF_Band", "EEG_TF_BandChan", "ACCEL_TF_Band", "ACCEL_TF_BandChan", "GYRO_TF_Band", "GYRO_TF_BandChan", "PPG_TF_Band", "PPG_TF_BandChan"]:
                            rect_x.right, rect_x.top, rect_x.bottom = 0, rect_buffer_x.top, rect_buffer_x.bottom
                            if rect_x.left > 30:     # stop pan out
                                rect_x.left = 30
                            if rect_x.left < 3:     # stop pan in
                                rect_x.left = 3
                            self.view_x_axis.camera.rect = rect_x
                            rect.right, rect.top, rect.bottom = window_buffer, rect_buffer.top, rect_buffer.bottom
                            if rect.left > rect_left_limit:     # stop pan in
                                rect.left = rect_left_limit
                            if rect.left < 0:                   # stop pan out
                                rect.left = 0
                            self.view[channel].camera.rect = rect
                        # === FFT Mouse_Move_Event: scale RIGHT, left = 0, top/bottom = fixed
                        if self.toggle_device_view in ["EEG_FFT", "ACCEL_FFT", "GYRO_FFT", "PPG_FFT"]:
                            # control for pan too far: set at 3, with window_eeg_fft = 128 this is 3 Hz
                            rect_x.left, rect_x.top, rect_x.bottom = 0, rect_buffer_x.top, rect_buffer_x.bottom
                            if self.toggle_device_view == "PPG_FFT":
                                if rect_x.right > (window_buffer / 10):            # stop pan out, FFT ONLY DIVIDE BY 10 becuase we are increasing frequency resolution
                                    rect_x.right = (window_buffer / 10)
                            else:
                                if rect_x.right > window_buffer:            # stop pan out
                                    rect_x.right = window_buffer
                            if rect_x.right < 3:                        # stop pan in
                                rect_x.right = 3
                            self.view_x_axis.camera.rect = rect_x
                            rect.left, rect.top, rect.bottom = 0, rect_buffer.top, rect_buffer.bottom
                            if rect.right > window_buffer:
                                rect.right = window_buffer
                            if rect.right < 3:
                                rect.right = 3
                            self.view[channel].camera.rect = rect
                    event.handled = True
                else:
                    event.handled = False
            # === Mouse_Press_Event
            elif event.type == 'mouse_press':
##                print('Mouse_Press_Event')
                # accept the event if it is button 1 or 2: This is required in order to receive future events
                event.handled = event.button in [1, 2]
            else:
                event.handled = False

######        # === Connect Keyboard/Mouse Events
######        self.canvas.events.key_press.connect(self.on_key_event)
######        self.canvas.events.mouse_press.connect(self.on_mouse_press)
######        self.canvas.events.mouse_move.connect(self.on_mouse_move)
######        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
######
######    # === Define Keyboard/Mouse Events
######    def on_key_event(self, event):
######        print('Key Event: ' + str(event.key))
######    def on_mouse_press(self, event):
######        print('Mouse_Press Event: ' + str(event.button))
######    def on_mouse_move(self, event):
######        print('Mouse_Move Event: ' + str(event.pos))
######    def on_mouse_wheel(self, event):
######        print('Mouse_Wheel Event: ' + str(event.delta))



##    def link_cam(self):
##        self.x_axis.link_view(self.view_x_axis)






    # CONNECT from DataSource pyqt to Canvas, includes everything from MyMainWindow pyqt in data_dict
    def update_data(self, new_data_dict):

        self.play_bool = new_data_dict["play_bool"]
        color_list = [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]]             # color_list = [black, red, blue, green, purple]
        band_range_eeg, band_range_accel, band_range_gyro, band_range_ppg = eeg_band_list_final, accel_band_list_final, gyro_band_list_final, ppg_band_list_final  

        # REFACTOR first get the device n_chan and n_window sampling_rate, add zoom factors, then paramaters for mouse camera control: zoom, rect_top_limit, rect_left_limit

        # MOVED VARS
        cam_window_eeg_tf = int(window_eeg / 12)
        cam_window_eeg_fft = 128

        cam_window_accel_tf = int(window_ppg / 4)
        cam_window_gyro_tf = int(window_ppg / 4)
        cam_window_ppg_tf = int(window_ppg / 4)

        cam_window_accel_fft = 26
        cam_window_gyro_fft = 26
        cam_window_ppg_fft = int((sampling_rate_ppg * 10) / 2)


        camera_param_dict = {
                     "EEG_Time_Series":         [n_chan_eeg,      window_eeg,       sampling_rate_eeg,   band_range_eeg,   0.2,  window_eeg],
                     "ACCEL_Time_Series":       [n_chan_accel,    window_accel,     sampling_rate_accel, band_range_accel, 0.2,  window_accel],
                     "GYRO_Time_Series":        [n_chan_gyro,     window_gyro,      sampling_rate_gyro,  band_range_gyro,  0.2,  window_gyro],
                     "PPG_Time_Series":         [n_chan_ppg,      window_ppg,       sampling_rate_ppg,   band_range_ppg,   0.2,  window_ppg],
                     "EEG_FFT":                 [n_chan_eeg_fft,  window_eeg_fft,   sampling_rate_eeg,   band_range_eeg,   0.05, window_eeg_fft - 1 - int(window_eeg_fft / 2)],
                     "ACCEL_FFT":               [n_chan_accel_fft,window_accel_fft, sampling_rate_accel, band_range_accel, 0.05, window_accel_fft - 1],
                     "GYRO_FFT":                [n_chan_gyro_fft, window_gyro_fft,  sampling_rate_gyro,  band_range_gyro,  0.05, window_gyro_fft - 1],
                     "PPG_FFT":                 [n_chan_ppg_fft,  window_ppg_fft,   sampling_rate_ppg,   band_range_ppg,   0.05, window_ppg_fft - 1],
                     "EEG_TF_BandChan":         [n_chan_eeg_tf,   window_eeg_tf,    sampling_rate_eeg,   band_range_eeg,   0.2,  window_eeg_tf],
                     "ACCEL_TF_BandChan":       [n_chan_accel_tf, window_accel_tf,  sampling_rate_accel, band_range_accel, 0.2,  window_accel_tf],
                     "GYRO_TF_BandChan":        [n_chan_gyro_tf,  window_gyro_tf,   sampling_rate_gyro,  band_range_gyro,  0.2,  window_gyro_tf],
                     "PPG_TF_BandChan":         [n_chan_ppg_tf,   window_ppg_tf,    sampling_rate_ppg,   band_range_ppg,   0.2,  window_ppg_tf],
                     "EEG_TF_Band":             [1,               window_eeg_tf,    sampling_rate_eeg,   band_range_eeg,   0.05, window_eeg_tf],
                     "ACCEL_TF_Band":           [1,               window_accel_tf,  sampling_rate_accel, band_range_accel, 0.05, window_accel_tf],
                     "GYRO_TF_Band":            [1,               window_gyro_tf,   sampling_rate_gyro,  band_range_gyro,  0.05, window_gyro_tf],
                     "PPG_TF_Band":             [1,               window_ppg_tf,    sampling_rate_ppg,   band_range_ppg,   0.05, window_ppg_tf],
                     "EEG_PSD":                 [n_chan_eeg,      window_eeg,       sampling_rate_eeg,   band_range_eeg,   0.2,  window_eeg], 
                     "ACCEL_PSD":               [n_chan_accel,    window_accel,     sampling_rate_accel, band_range_accel, 0.2,  window_accel], 
                     "GYRO_PSD":                [n_chan_gyro,     window_gyro,      sampling_rate_gyro,  band_range_gyro,  0.2,  window_gyro], 
                     "PPG_PSD":                 [n_chan_ppg,      window_ppg,       sampling_rate_ppg,   band_range_ppg,   0.2,  window_ppg]}

        n_chan_buffer = camera_param_dict[str(new_data_dict["toggle_view"])][0]
        window_buffer = camera_param_dict[str(new_data_dict["toggle_view"])][1]
        sampling_rate_buffer = camera_param_dict[str(new_data_dict["toggle_view"])][2]
        band_range_buffer = camera_param_dict[str(new_data_dict["toggle_view"])][3]
        stretch_x_axis_buffer = camera_param_dict[str(new_data_dict["toggle_view"])][4]
        cam_window_buffer = camera_param_dict[str(new_data_dict["toggle_view"])][5]


        # UPDATE DATA
        if str(self.toggle_device_view) != str(new_data_dict["toggle_view"]):


            self.update_x_axis = True


            # reset self.cam vars
            self.cam_toggle, self.current_cam_toggle_canvas, self.last_cam_toggle_canvas = "Auto", "Auto", "Auto"

            # reset canvas variables
            self.canvas.central_widget.remove_widget(self.grid)
            self.grid = self.canvas.central_widget.add_grid()
            self.view, self.line, self.y_axis_all = [], [], []
            print('current_view: ' + str(self.toggle_device_view) + ', new_view: ' + str(new_data_dict["toggle_view"]))

            # === PSD all device
            if str(new_data_dict["toggle_view"]) in ["EEG_PSD", "ACCEL_PSD", "GYRO_PSD", "PPG_PSD"]:
                data_psd_dict = {"EEG_PSD": [freqNum_psd_eeg, IMAGE_SHAPE], "ACCEL_PSD": [freqNum_psd_accel, IMAGE_SHAPE_accel], "GYRO_PSD": [freqNum_psd_gyro, IMAGE_SHAPE_gyro], "PPG_PSD": [freqNum_psd_ppg, IMAGE_SHAPE_ppg]}
                freqNum_psd_buffer, IMAGE_SHAPE_buffer = data_psd_dict[str(new_data_dict["toggle_view"])][0], data_psd_dict[str(new_data_dict["toggle_view"])][1]
                self.image, self.psd_colorbar_all = [], []
                image_data_buffer = np.zeros([freqNum_psd_buffer, cam_window_buffer])
##                image_data_buffer = np.zeros([freqNum_psd_buffer, window_buffer])
                for channel in range(0, n_chan_buffer):
                    self.view.append(self.grid.add_view(channel, 0, bgcolor='cyan'))
                    self.image.append(visuals.Image(image_data_buffer, texture_format="auto", cmap=COLORMAP_CHOICES[0], parent=self.view[channel].scene))
                    self.view[channel].camera = "panzoom"
                    self.view[channel].camera.set_range(x=(0, IMAGE_SHAPE_buffer[1]), y=(0, IMAGE_SHAPE_buffer[0]), margin=0)
                    self.view[channel].camera.rect.left = int(window_buffer / 2)
##                    # define rect and disconnect mouse events from view[channel].camera
##                    self.view[channel].camera = scene.PanZoomCamera(rect=(n_rect_pos[0], n_rect_pos[1], cam_window_buffer, n_rect_pos[3]), interactive=True)
####                    self.view[channel].camera.rect.left, self.view[channel].camera.rect.right = int(cam_window_buffer / 2), cam_window_buffer
##                    self.view[channel].camera.rect.right = int(window_buffer / 2)
                    # === VIEW disconnect events
                    self.view[channel].camera._viewbox.events.mouse_move.disconnect(self.view[channel].camera.viewbox_mouse_event)
                    self.view[channel].camera._viewbox.events.mouse_press.disconnect(self.view[channel].camera.viewbox_mouse_event)
                    self.view[channel].camera._viewbox.events.mouse_wheel.disconnect(self.view[channel].camera.viewbox_mouse_event)
                    # === SELF.Y_AXIS VIEW
##                    self.y_axis_all.append(scene.AxisWidget(orientation='right', axis_label='Hz', axis_label_margin=-1))
                    self.y_axis_all.append(scene.AxisWidget(orientation='right', axis_label='Hz', axis_label_margin=-1, axis_color='black', text_color='black'))
                    self.view[channel].add_widget(self.y_axis_all[channel])
                    self.y_axis_all[channel].link_view(self.view[channel])
                # === define top/bot camera view left/right limits
                self.psd_scale = 1
                rect_buffer_psd_original = Rect(self.view[0].camera.rect)
                self.view_psd_left_limit, self.view_psd_right_limit = 0, rect_buffer_psd_original.right
####                self.view_psd_left_limit, self.view_psd_right_limit = rect_buffer_psd_original.left, rect_buffer_psd_original.right
##                print('self.view_psd_left_limit, self.view_psd_right_limit: ' + str([rect_buffer_psd_original.left, rect_buffer_psd_original.right]))

            # === FFT all device
            elif str(new_data_dict["toggle_view"]) in ["EEG_FFT", "ACCEL_FFT", "GYRO_FFT", "PPG_FFT"]:
                self.view.append(self.grid.add_view(0, 0, bgcolor='#c0c0c0'))
                line_data_buffer = np.swapaxes(np.asarray([np.arange(window_buffer), np.zeros(window_buffer)]), 0, 1)
##                line_data_buffer = np.swapaxes(np.asarray([np.arange(cam_window_buffer), np.zeros(cam_window_buffer)]), 0, 1)
                for channel in range(0, n_chan_buffer):
                    self.line.append(visuals.Line(line_data_buffer, parent=self.view[0].scene, color=LINE_COLOR_CHOICES[channel]))
                self.view[0].camera = scene.PanZoomCamera(rect=(0, -.01, cam_window_buffer, 25), interactive=True)
                # === VIEW disconnect events
                self.view[0].camera._viewbox.events.mouse_move.disconnect(self.view[0].camera.viewbox_mouse_event)
                self.view[0].camera._viewbox.events.mouse_press.disconnect(self.view[0].camera.viewbox_mouse_event)
                self.view[0].camera._viewbox.events.mouse_wheel.disconnect(self.view[0].camera.viewbox_mouse_event)
                # === SELF.Y_AXIS VIEW
##                self.y_axis_all.append(scene.AxisWidget(orientation='right', axis_label='dB', axis_label_margin=-1))
                self.y_axis_all.append(scene.AxisWidget(orientation='right', axis_label='dB', axis_label_margin=-1, axis_color='black', text_color='black'))
                self.view[0].add_widget(self.y_axis_all[0])
                self.y_axis_all[0].link_view(self.view[0])



            # === TF Single/Multi Plot: Both Separate Band, Only Multi = Separate Channel
            elif str(new_data_dict["toggle_view"]) in ["EEG_TF_BandChan", "EEG_TF_Band", "ACCEL_TF_BandChan", "ACCEL_TF_Band", "GYRO_TF_BandChan", "GYRO_TF_Band", "PPG_TF_BandChan", "PPG_TF_Band"]:
                # ========= TF Single/Multi Plot: COLOR =========
                self.line_color = []
                line_data_buffer = np.swapaxes(np.asarray([np.arange(cam_window_buffer), np.zeros(cam_window_buffer)]), 0, 1)
                for channel in range(0, n_chan_buffer):
                    self.view.append(self.grid.add_view(channel, 0, bgcolor='#c0c0c0'))
                    for band in range(0, len(band_range_buffer)):
                        color = np.ones((cam_window_buffer, 4), dtype=np.float32)
##                        color = np.ones((window_buffer, 4), dtype=np.float32)
                        color[:, 0:4] = color_list[band]
                        self.line_color.append(color)
                        self.line.append(visuals.Line(line_data_buffer, parent=self.view[channel].scene, color=color))
                    # define rect and disconnect mouse events from view[channel].camera
                    self.view[channel].camera = scene.PanZoomCamera(rect=(0, -.01, cam_window_buffer, 25), interactive=True)
                    self.view[channel].camera.rect.left, self.view[channel].camera.rect.right = int(cam_window_buffer / 2), cam_window_buffer
                    self.view[channel].camera._viewbox.events.mouse_move.disconnect(self.view[channel].camera.viewbox_mouse_event)
                    self.view[channel].camera._viewbox.events.mouse_press.disconnect(self.view[channel].camera.viewbox_mouse_event)
                    self.view[channel].camera._viewbox.events.mouse_wheel.disconnect(self.view[channel].camera.viewbox_mouse_event)
                    # === SELF.Y_AXIS VIEW
##                    self.y_axis_all.append(scene.AxisWidget(orientation='right', axis_label='dB', axis_label_margin=-1))
                    self.y_axis_all.append(scene.AxisWidget(orientation='right', axis_label='dB', axis_label_margin=-1, axis_color='black', text_color='black'))
                    self.view[channel].add_widget(self.y_axis_all[channel])
                    self.y_axis_all[channel].link_view(self.view[channel])
                # === update self.line_color_plot for bandchan
                self.line_color_plot = []
                for filt in range(0, len(self.eeg_filter_list_range) + 1):
                    self.line_color_plot.append(self.line_color)
            # ========= TF Single/Multi Plot: COLOR =========

            else:
                # ========= TIME SERIES: COLOR VIEW CAMERA UPDATED NEW =========
                self.line_color = []
##                line_data_buffer = np.swapaxes(np.asarray([np.arange(window_buffer), np.zeros(window_buffer)]), 0, 1)
                line_data_buffer = np.swapaxes(np.asarray([np.arange(cam_window_buffer), np.zeros(cam_window_buffer)]), 0, 1)
                for channel in range(0, n_chan_buffer):
                    color = np.ones((cam_window_buffer, 4), dtype=np.float32)
##                    color = np.ones((window_buffer, 4), dtype=np.float32)
                    color[:, 0:4] = color_list[channel]
                    self.line_color.append(color)
                    self.view.append(self.grid.add_view(channel, 0, bgcolor='#c0c0c0'))
                    self.line.append(visuals.Line(line_data_buffer, parent=self.view[channel].scene, color=color))
                    self.view[channel].camera = scene.PanZoomCamera(rect=(0, -25, window_buffer, 50), interactive=True)
                    self.view[channel].camera.rect.left, self.view[channel].camera.rect.right = int(window_buffer / 2), window_buffer
                    # === DISCONNECT MOUSE EVENTS from auto control from vispy so you can manually edit below
                    self.view[channel].camera._viewbox.events.mouse_move.disconnect(self.view[channel].camera.viewbox_mouse_event)
                    self.view[channel].camera._viewbox.events.mouse_press.disconnect(self.view[channel].camera.viewbox_mouse_event)
                    self.view[channel].camera._viewbox.events.mouse_wheel.disconnect(self.view[channel].camera.viewbox_mouse_event)
                    # === SELF.Y_AXIS VIEW WORKING BASIC
##                    self.y_axis_all.append(scene.AxisWidget(orientation='right', axis_label='uV', axis_label_margin=-1))
                    self.y_axis_all.append(scene.AxisWidget(orientation='right', axis_label='uV', axis_label_margin=-1, axis_color='black', text_color='black'))
                    self.view[channel].add_widget(self.y_axis_all[channel])
                    self.y_axis_all[channel].link_view(self.view[channel])
                # CONNECT MOUSE EVENTS to viewbox_mouse_event for manual scaling with mouse: self.canvas.events = mouse_wheel, mouse_press, mouse_move
                self.canvas.events.mouse_press.connect(self.viewbox_mouse_event)
                self.canvas.events.mouse_move.connect(self.viewbox_mouse_event)
                self.canvas.events.mouse_wheel.connect(self.viewbox_mouse_event)
                # === update self.line_color_plot for timeseries
                self.line_color_plot = []
                for filt in range(0, len(self.eeg_filter_list_range) + 1):
                    self.line_color_plot.append(self.line_color)






            # === SELF.X_AXIS VIEW ALL VIEW
            self.x_axis._parent = None
            self.view_x_axis = self.grid.add_view(len(self.view), 0, bgcolor='black')
            self.view_x_axis.camera = "panzoom"
            # === FFT
            if str(new_data_dict["toggle_view"]) in ["EEG_FFT", "ACCEL_FFT", "GYRO_FFT", "PPG_FFT"]:
####                self.view_x_axis.camera.set_range(x=(0, window_buffer), y=(-1, 1), margin=0)
                self.view_x_axis.camera.set_range(x=(0, cam_window_buffer), y=(-1, 1), margin=0)
####                self.view_x_axis.camera.rect.right, self.view_x_axis.camera.rect.left = int(cam_window_buffer), 0
            else:
                self.view_x_axis.camera.set_range(x=(int(cam_window_buffer / sampling_rate_buffer), 0), y=(-1, 1), margin=0)
##                self.view_x_axis.camera.set_range(x=(int(window_buffer / sampling_rate_buffer), 0), y=(-1, 1), margin=0)
            # SELF.X_AXIS WIDGET
            self.x_axis = scene.AxisWidget(orientation='bottom')
            if str(new_data_dict["toggle_view"]) != "EEG_FFT" and str(new_data_dict["toggle_view"]) != "ACCEL_FFT" and str(new_data_dict["toggle_view"]) != "GYRO_FFT" and str(new_data_dict["toggle_view"]) != "PPG_FFT":
                domain_list = list(self.x_axis.axis._domain)
                domain_list[0], domain_list[1] = int((window_buffer / 2) / sampling_rate_buffer), 0
                self.x_axis.axis._domain = tuple(domain_list)
            # === TIME_SERIES/PSD: SELF.X_AXIS VIEW
            if str(new_data_dict["toggle_view"]) in ["EEG_Time_Series", "ACCEL_Time_Series", "GYRO_Time_Series", "PPG_Time_Series", "EEG_PSD", "ACCEL_PSD", "GYRO_PSD", "PPG_PSD"]:
                self.view_x_axis.camera.rect.left, self.view_x_axis.camera.rect.right = int((window_buffer / 2) / sampling_rate_buffer), 0
##            # === FFT
##            if str(new_data_dict["toggle_view"]) == "EEG_FFT" or str(new_data_dict["toggle_view"]) == "ACCEL_FFT" or str(new_data_dict["toggle_view"]) == "GYRO_FFT" or str(new_data_dict["toggle_view"]) == "PPG_FFT":
##                self.view_x_axis.camera.rect.right, self.view_x_axis.camera.rect.left = int(cam_window_buffer), 0
            # === TF_BAND AND BANDCHAN: SELF.X_AXIS VIEW
            if str(new_data_dict["toggle_view"]) in ["EEG_TF_BandChan", "EEG_TF_Band", "ACCEL_TF_BandChan", "ACCEL_TF_Band", "GYRO_TF_BandChan", "GYRO_TF_Band", "PPG_TF_BandChan", "PPG_TF_Band"]:
                # CHANGE X AXIS PSD FROM WINDOW_EEG TO CAM_WINDOW_EEG IF DESIRED HERE
                if str(new_data_dict["toggle_view"]) in ["EEG_TF_BandChan", "EEG_TF_Band"]:
                    self.view_x_axis.camera.rect.left, self.view_x_axis.camera.rect.right = int(((cam_window_buffer / 2) * 12) / sampling_rate_buffer), 0
                else:
                    self.view_x_axis.camera.rect.left, self.view_x_axis.camera.rect.right = int(((cam_window_buffer / 2) * 4) / sampling_rate_buffer), 0
            # === Add Widget VIEW SELF.X_AXIX disconnect/connect mouse events
            self.x_axis.stretch = (1, stretch_x_axis_buffer)
            self.grid.add_widget(self.x_axis, row=len(self.view), col=0)
            self.view_x_axis.camera._viewbox.events.mouse_move.disconnect(self.view_x_axis.camera.viewbox_mouse_event)
            self.view_x_axis.camera._viewbox.events.mouse_press.disconnect(self.view_x_axis.camera.viewbox_mouse_event)
            self.view_x_axis.camera._viewbox.events.mouse_wheel.disconnect(self.view_x_axis.camera.viewbox_mouse_event)
            # define limits
            rect_buffer_x_axis_original = Rect(self.view_x_axis.camera.rect)
            # === FFT
            if str(new_data_dict["toggle_view"]) in ["EEG_FFT", "ACCEL_FFT", "GYRO_FFT", "PPG_FFT"]:
                if str(new_data_dict["toggle_view"]) == "EEG_FFT":
                    self.view_x_axis_right_limit = int(rect_buffer_x_axis_original.right * 2)
                if str(new_data_dict["toggle_view"]) == "PPG_FFT":
##                    self.view_x_axis_right_limit = int(rect_buffer_x_axis_original.right)
                    rect_buffer_x_axis_original.right = rect_buffer_x_axis_original.right / 10
                    self.view_x_axis_right_limit = int(rect_buffer_x_axis_original.right)
                    self.view_x_axis.camera.rect = rect_buffer_x_axis_original
                else:
                    self.view_x_axis_right_limit = rect_buffer_x_axis_original.right
            # === TIME_SERIES/PSD
            else:
                self.view_x_axis_left_limit, self.view_x_axis_right_limit = int(rect_buffer_x_axis_original.left * 2), rect_buffer_x_axis_original.right






        # ========= UPDATE DATA OG =========
        self.line_bool_canvas = new_data_dict["line_toggles"]
        self.toggle_device_view = new_data_dict["toggle_view"]
        # === PSD IMAGE UPDATE TREE
        if self.toggle_device_view in ["EEG_PSD", "ACCEL_PSD", "GYRO_PSD", "PPG_PSD"]:
            psd_data_buffer = np.asarray(new_data_dict[str(self.toggle_device_view)])
            # === PSD PLOT UPDATE AND AUTO SCALE Z AXIS: update cmap/clim for colormap/limits
            if self.play_bool == True:
                for channel in range(0, len(psd_data_buffer)):
                    if self.line_bool_canvas[channel] == True:
                        self.image[channel].set_data(psd_data_buffer[channel])
                    else:
##                    if self.line_bool_canvas[channel] == False:
                        data_psd_dict = {"EEG_PSD": [freqNum_psd_eeg, IMAGE_SHAPE], "ACCEL_PSD": [freqNum_psd_accel, IMAGE_SHAPE_accel], "GYRO_PSD": [freqNum_psd_gyro, IMAGE_SHAPE_gyro], "PPG_PSD": [freqNum_psd_ppg, IMAGE_SHAPE_ppg]}
                        freqNum_psd_buffer = data_psd_dict[str(new_data_dict["toggle_view"])][0]
                        image_data_buffer = np.zeros([freqNum_psd_buffer, cam_window_buffer])
                        self.image[channel].set_data(image_data_buffer)

##                    self.image[channel].set_data(psd_data_buffer[channel])
    ####                data_buffer = np.array(psd_data_buffer[channel])
    ####                # === MORLET SCALING using clim_list: LONG DATA WINDOW SCALING: note: auto/lock camera will override, but this line of code is necessary or psd plot stops moving
    ####                clim_list[0] = np.median(data_buffer) - ((np.std(data_buffer) * self.psd_scale))
    ####                clim_list[1] = np.median(data_buffer) + ((np.std(data_buffer) * 3) * self.psd_scale)
    ####                self.image[channel].clim = tuple(clim_list)
    ##            print('device_view: ' + str(self.toggle_device_view) + ', time_diff: ' + str(local_clock() - self.last_time_test))
    ##            self.last_time_test = local_clock()

        # EEG_Time_Series/FFT: 4 channels, ACCEL/GYRO/PPG_Time_Series/FFT: 3 channels
        elif self.toggle_device_view in ["EEG_Time_Series", "EEG_FFT", "ACCEL_Time_Series", "GYRO_Time_Series", "PPG_Time_Series", "ACCEL_FFT", "GYRO_FFT", "PPG_FFT", "ACCEL_TF_Band", "GYRO_TF_Band", "PPG_TF_Band"]:
            line_data_buffer = np.asarray(new_data_dict[str(self.toggle_device_view)])
            if self.play_bool == True:
                for line in range(0, len(line_data_buffer)):
                    if self.line_bool_canvas[line] == False:
                        line_data_buffer[line, :, 1] = 0
                        self.line[line].set_data(line_data_buffer[line], color='#c0c0c0')
                    else:
                        if self.toggle_device_view == "EEG_Time_Series":
                            self.line[line].set_data(line_data_buffer[line], color=self.line_color_plot[self.current_artifact_filt][line])
                        elif self.toggle_device_view == "ACCEL_Time_Series":
                            self.line[line].set_data(line_data_buffer[line], color=self.line_color_plot_accel[self.current_artifact_filt_accel][line])
                        elif self.toggle_device_view == "GYRO_Time_Series":
                            self.line[line].set_data(line_data_buffer[line], color=self.line_color_plot_gyro[self.current_artifact_filt_gyro][line])
                        else:
                            self.line[line].set_data(line_data_buffer[line], color=LINE_COLOR_CHOICES[line])

        # === BANDPOWER
        elif self.toggle_device_view in ["EEG_TF_BandChan", "EEG_TF_Band"]:
            line_data_buffer = np.asarray(new_data_dict[str(self.toggle_device_view)])
            if new_data_dict["toggle_view"] == "EEG_TF_BandChan":
                if self.play_bool == True:
                    for channel in range(0, n_chan_eeg):
                        for band in range(0, len(band_range_eeg)):
                            current_line = int(band + (channel * (n_chan_eeg + 1)))
                            if self.line_bool_canvas[band] == False:
                                line_data_buffer[current_line, :, 1] = 0
                                self.line[current_line].set_data(line_data_buffer[current_line], color='#c0c0c0')
                            else:
                                self.line[current_line].set_data(line_data_buffer[current_line], color=self.line_color_plot[self.current_artifact_filt][current_line])
##                    print('device_view: ' + str(self.toggle_device_view) + ', time_diff: ' + str(local_clock() - self.last_time_test))
##                    self.last_time_test = local_clock()
            if self.toggle_device_view == "EEG_TF_Band":
                if self.play_bool == True:
                    for line in range(0, len(line_data_buffer)):
    ##                for band in range(0, len(band_range_eeg)):
                        if self.line_bool_canvas[line] == False:
                            line_data_buffer[line, :, 1] = 0
                            self.line[line].set_data(line_data_buffer[line], color='#c0c0c0')
                        else:
                            self.line[line].set_data(line_data_buffer[line], color=self.line_color_plot[self.current_artifact_filt][line])
##                    print('device_view: ' + str(self.toggle_device_view) + ', time_diff: ' + str(local_clock() - self.last_time_test))
##                    self.last_time_test = local_clock()
        # BAND/BAND_CHAN: 5 BANDS: EEG_TF_BandChan: 4 channels, ACCEL/GYRO/PPG_TF_BandChan: 3 channels, EEG/ACCEL/GYRO/PPG_TF_Band: 1 channels
        elif self.toggle_device_view in ["ACCEL_TF_BandChan", "GYRO_TF_BandChan", "PPG_TF_BandChan"]:
            line_data_buffer = np.asarray(new_data_dict[str(self.toggle_device_view)])
##            print('line_data_buffer.shape: ' + str(np.array(line_data_buffer).shape))
            if self.toggle_device_view in ["ACCEL_TF_BandChan", "GYRO_TF_BandChan", "PPG_TF_BandChan"]:
                if self.play_bool == True:
                    current_line = 0
                    for channel in range(0, 3):
                        for band in range(0, 5):
                            if self.line_bool_canvas[band] == False:
                                line_data_buffer[current_line, :, 1] = 0
                                self.line[current_line].set_data(line_data_buffer[current_line], color='#c0c0c0')
                            else:
                                self.line[current_line].set_data(line_data_buffer[current_line], color=self.line_color_plot[self.current_artifact_filt][current_line])
                            current_line += 1
##                    print('device_view: ' + str(self.toggle_device_view) + ', time_diff: ' + str(local_clock() - self.last_time_test))
##                    self.last_time_test = local_clock()

##        else:
##            line_data_buffer = np.asarray(new_data_dict[str(self.toggle_device_view)])
##            for line in range(0, len(line_data_buffer)):
##                self.line[line].set_data(line_data_buffer[line])
####            self.last_time_test = local_clock()
        # ========= UPDATE DATA NEW =========





##        # ========= ARTIFACT LINES FILTER LINE CORRECT NEW =========
##        timestamps_eeg_len_buffer = new_data_dict["timestamps_eeg_len"]
##        artifact_pos_buffer = new_data_dict["artifact_pos_all"]
##        if self.current_artifact_filt != new_data_dict["current_artifact_filt_eeg"]:
##            # define last filter and current filter difference to add to current pos for lag correction
##            bf_eeg_index_offset = new_data_dict["bf_eeg_index_offset"]
##            filter_offset_diff = bf_eeg_index_offset[new_data_dict["current_artifact_filt_eeg"]] - bf_eeg_index_offset[self.current_artifact_filt]
##            print('=== CORRECTING FILT: self.current_artifact_filt: ' + str(self.current_artifact_filt) + ', new_data_dict["current_artifact_filt_eeg"]: ' + str(new_data_dict["current_artifact_filt_eeg"])
##                  + ', filter_offset_diff: ' + str(filter_offset_diff))
##            # set artifact line data
##            for channel in range(0, len(self.marker_line_all)):
##                if len(self.marker_line_all[channel]) > 0:
##                    for this_line in range(0, len(self.marker_line_all[channel])):
##                        self.marker_line_all[channel][this_line].set_data(self.marker_line_all[channel][this_line].pos + filter_offset_diff)
##            self.current_artifact_filt = new_data_dict["current_artifact_filt_eeg"]
##        # === ARTIFACT LINES
##        # remove old artifact lines
##        for channel in range(0, len(self.marker_line_all)):
##            remove_line_buffer = 0
##            for this_line in range(0, len(self.marker_line_all[channel])):
##                if (self.marker_line_all[channel][this_line].pos - timestamps_eeg_len_buffer) < (0 - self.window_eeg_long):
##                    remove_line_buffer += 1
##            if remove_line_buffer > 0:
##                for this_marker in range(0, remove_line_buffer):
##                    self.marker_line_all[channel][this_marker].parent = None
##                self.marker_line_all[channel] = self.marker_line_all[channel][remove_line_buffer:]
##        # === ARTIFACT: update artifact lines
##        for channel in range(0, len(self.marker_line_all)):
##            # set artifact line data
##            if len(self.marker_line_all[channel]) > 0:
##                for this_line in range(0, len(self.marker_line_all[channel])):
##                    self.marker_line_all[channel][this_line].set_data(self.marker_line_all[channel][this_line].pos - timestamps_eeg_len_buffer)
##            # append new artifact lines
##            if len(artifact_pos_buffer[channel]) > 0:
##                for this_line in range(0, len(artifact_pos_buffer[channel])):
####                    if new_data_dict["bf_eeg_index_offset"][self.current_artifact_filt]
##                    self.marker_line_all[channel].append(scene.InfiniteLine(artifact_pos_buffer[channel][this_line] + new_data_dict["bf_eeg_index_offset"][self.current_artifact_filt], [1.0, 0.1, 0.8, 1.0], vertical=True, parent=self.view[channel].scene))         # teal
####                    self.marker_line_all[channel].append(scene.InfiniteLine(artifact_pos_buffer[channel][this_line], [1.0, 0.1, 0.8, 1.0], vertical=True, parent=self.view[channel].scene))         # teal
##                    self.marker_line_all[channel][len(self.marker_line_all[channel]) - 1]._opacity = .01
####        print('self.marker_line_values: ' + str(self.marker_line_values))
##        # ========= FILTER LINE CORRECT NEW =========














        # ========= EEG ARTIFACT TIME SERIES LINE COLOR UPDATE NEW =========
        if new_data_dict["toggle_view"] == "EEG_Time_Series":
            # RESET COLOR if we change to artifact_off
            if new_data_dict["corner_widget_bool"] != self.corner_widget_bool_canvas:
                if new_data_dict["corner_widget_bool"] == False:
                    self.line_color_plot = []
                    for filt in range(0, len(self.eeg_filter_list_range) + 1):
                        self.line_color_plot.append(self.line_color)
                    self.line_color_long = []
                    for channel in range(0, n_chan_eeg):
                        color = np.ones((self.window_eeg_long, 4), dtype=np.float32)
                        color[:, 0:4] = color_list[channel]
                        self.line_color_long.append(color)
                # update self.corner_widget_bool
                self.corner_widget_bool_canvas = new_data_dict["corner_widget_bool"]
            # === ARTIFACT UPDATE COLOR
            if new_data_dict["corner_widget_bool"] == True:
                artifact_pos_buffer = new_data_dict["artifact_pos_all"]
                timestamps_eeg_len_buffer = new_data_dict["timestamps_len"]
                offset_buffer = self.window_eeg_long - window_eeg
                self.line_color_long = np.roll(self.line_color_long, -timestamps_eeg_len_buffer, axis=1)
                self.line_color_plot = np.roll(self.line_color_plot, -timestamps_eeg_len_buffer, axis=2)
                for channel in range(0, len(self.line_color_long)):
        ##            self.line_color_long[channel, -timestamps_eeg_len_buffer:self.window_eeg_long, 0:4] = color_list[channel]
                    self.line_color_long[channel, (self.window_eeg_long-timestamps_eeg_len_buffer):self.window_eeg_long, 0:4] = color_list[channel]
                # update data
                self.current_artifact_filt = new_data_dict["current_artifact_filt_eeg"]
                # if no new artifact detection performed, simply update line_color_plot with line_color_long
                if self.current_artifact_pos == artifact_pos_buffer:
                    for channel in range(0, n_chan_eeg):
                        for filt in range(0, len(self.line_color_plot)):
                            self.line_color_plot[filt, channel, :, 0:4] = self.line_color_long[channel, self.line_color_range[filt][0]:self.line_color_range[filt][1], 0:4]
        ##                    self.line_color_plot[filt, channel, :, 0:4] = self.line_color_long[channel, self.line_color_range[filt][0] - offset_buffer:self.line_color_range[filt][1] - offset_buffer, 0:4]
                else:
        ##        if self.current_artifact_pos != artifact_pos_buffer:
##                    if new_data_dict["artifact_toggle"] == "artifact_mask":
                    offset_buffer = self.window_eeg_long - window_eeg
                    color_artifact = [1.0, 1.0, 1.0, 1.0]           # white
                    for channel in range(0, n_chan_eeg):
                        for artifact in range(0, len(artifact_pos_buffer[channel]), 2):
                            self.line_color_long[channel, (artifact_pos_buffer[channel][artifact] + offset_buffer):(artifact_pos_buffer[channel][artifact + 1] + offset_buffer), 0:4] = color_artifact
                            # update self.line_color_plot with new artifacts using long data
                            for filt in range(0, len(self.line_color_plot)):
                                self.line_color_plot[filt, channel, :, 0:4] = self.line_color_long[channel, self.line_color_range[filt][0]:self.line_color_range[filt][1], 0:4]
                    self.current_artifact_pos = artifact_pos_buffer
        # ========= EEG ARTIFACT LINE COLOR UPDATE NEW =========



        # ========= ACCEL/GYRO ARTIFACT TIME SERIES LINE COLOR UPDATE NEW =========
        if new_data_dict["toggle_view"] in ["ACCEL_Time_Series", "GYRO_Time_Series"]:
            # RESET COLOR if we change to artifact_off
            if new_data_dict["corner_widget_bool"] != self.corner_widget_bool_canvas:
                if new_data_dict["corner_widget_bool"] == False:
                    if new_data_dict["toggle_view"] == "ACCEL_Time_Series":
                        self.line_color_plot_accel = []
                        for filt in range(0, len(self.accel_filter_list_range) + 1):
                            self.line_color_plot_accel.append(self.line_color_accel)
                        self.line_color_long_accel = []
                        for channel in range(0, n_chan_accel):
                            color = np.ones((self.window_accel_long, 4), dtype=np.float32)
                            color[:, 0:4] = color_list[channel]
                            self.line_color_long_accel.append(color)
                    if new_data_dict["toggle_view"] == "GYRO_Time_Series":
                        self.line_color_plot_gyro = []
                        for filt in range(0, len(self.gyro_filter_list_range) + 1):
                            self.line_color_plot_gyro.append(self.line_color_gyro)
                        self.line_color_long_gyro = []
                        for channel in range(0, n_chan_gyro):
                            color = np.ones((self.window_gyro_long, 4), dtype=np.float32)
                            color[:, 0:4] = color_list[channel]
                            self.line_color_long_gyro.append(color)
                # update self.corner_widget_bool
                self.corner_widget_bool_canvas = new_data_dict["corner_widget_bool"]
            # === ARTIFACT UPDATE COLOR
            if new_data_dict["corner_widget_bool"] == True:
                timestamps_accel_len_buffer = new_data_dict["timestamps_len"]
                if new_data_dict["toggle_view"] == "ACCEL_Time_Series":
                    artifact_pos_buffer_accel = new_data_dict["artifact_pos_all_accel"]
                    offset_buffer = self.window_accel_long - window_accel
                    self.line_color_long_accel = np.roll(self.line_color_long_accel, -timestamps_accel_len_buffer, axis=1)
                    self.line_color_plot_accel = np.roll(self.line_color_plot_accel, -timestamps_accel_len_buffer, axis=2)
                    for channel in range(0, len(self.line_color_long_accel)):
                        self.line_color_long_accel[channel, (self.window_accel_long-timestamps_accel_len_buffer):self.window_accel_long, 0:4] = color_list[channel]
                    # update data
                    self.current_artifact_filt_accel = new_data_dict["current_artifact_filt_accel"]
                    # if no new artifact detection performed, simply update line_color_plot with line_color_long
                    if self.current_artifact_pos_accel == artifact_pos_buffer_accel:
                        for channel in range(0, n_chan_accel):
                            for filt in range(0, len(self.line_color_plot_accel)):
                                self.line_color_plot_accel[filt, channel, :, 0:4] = self.line_color_long_accel[channel, self.line_color_range_accel[filt][0]:self.line_color_range_accel[filt][1], 0:4]
                    else:
                        offset_buffer = self.window_accel_long - window_accel
                        color_artifact = [1.0, 1.0, 1.0, 1.0]           # white
                        for channel in range(0, n_chan_accel):
                            for artifact in range(0, len(artifact_pos_buffer_accel[channel]), 2):
                                self.line_color_long_accel[channel, (artifact_pos_buffer_accel[channel][artifact] + offset_buffer):(artifact_pos_buffer_accel[channel][artifact + 1] + offset_buffer), 0:4] = color_artifact
                                # update self.line_color_plot with new artifacts using long data
                                for filt in range(0, len(self.line_color_plot_accel)):
                                    self.line_color_plot_accel[filt, channel, :, 0:4] = self.line_color_long_accel[channel, self.line_color_range_accel[filt][0]:self.line_color_range_accel[filt][1], 0:4]
                        self.current_artifact_pos_accel = artifact_pos_buffer_accel
                if new_data_dict["toggle_view"] == "GYRO_Time_Series":
                    artifact_pos_buffer_gyro = new_data_dict["artifact_pos_all_gyro"]
                    offset_buffer = self.window_gyro_long - window_gyro
                    self.line_color_long_gyro = np.roll(self.line_color_long_gyro, -timestamps_gyro_len_buffer, axis=1)
                    self.line_color_plot_gyro = np.roll(self.line_color_plot_gyro, -timestamps_gyro_len_buffer, axis=2)
                    for channel in range(0, len(self.line_color_long_gyro)):
                        self.line_color_long_gyro[channel, (self.window_gyro_long-timestamps_gyro_len_buffer):self.window_gyro_long, 0:4] = color_list[channel]
                    # update data
                    self.current_artifact_filt_gyro = new_data_dict["current_artifact_filt_gyro"]
                    # if no new artifact detection performed, simply update line_color_plot with line_color_long
                    if self.current_artifact_pos_gyro == artifact_pos_buffer_gyro:
                        for channel in range(0, n_chan_gyro):
                            for filt in range(0, len(self.line_color_plot_gyro)):
                                self.line_color_plot_gyro[filt, channel, :, 0:4] = self.line_color_long_gyro[channel, self.line_color_range_gyro[filt][0]:self.line_color_range_gyro[filt][1], 0:4]
                    else:
                        offset_buffer = self.window_gyro_long - window_gyro
                        color_artifact = [1.0, 1.0, 1.0, 1.0]           # white
                        for channel in range(0, n_chan_gyro):
                            for artifact in range(0, len(artifact_pos_buffer_gyro[channel]), 2):
                                self.line_color_long_gyro[channel, (artifact_pos_buffer_gyro[channel][artifact] + offset_buffer):(artifact_pos_buffer_gyro[channel][artifact + 1] + offset_buffer), 0:4] = color_artifact
                                # update self.line_color_plot with new artifacts using long data
                                for filt in range(0, len(self.line_color_plot_gyro)):
                                    self.line_color_plot_gyro[filt, channel, :, 0:4] = self.line_color_long_gyro[channel, self.line_color_range_gyro[filt][0]:self.line_color_range_gyro[filt][1], 0:4]
                        self.current_artifact_pos_gyro = artifact_pos_buffer_gyro
        # ========= ACCEL/GYRO ARTIFACT LINE COLOR UPDATE NEW =========




# ==================== REFACTOR CAMERA ====================

        # WHY IS N_CHAN_BUFFER USED HERE

        # === RESET CAMERA UPDATE
        self.current_cam_toggle_canvas = new_data_dict["cam_toggle"]
##        print('self.current_cam_toggle_canvas: ' + str(self.current_cam_toggle_canvas))

        # first check for camera click and update self.current_cam_toggle_canvas so we can track auto -> auto and reset -> reset clicks
        if self.cam_toggle_count != new_data_dict["cam_toggle_count"]:
            self.current_cam_toggle_canvas = "Unlock"
        self.cam_toggle_count = new_data_dict["cam_toggle_count"]

        if self.current_cam_toggle_canvas != self.last_cam_toggle_canvas:
            if new_data_dict["cam_toggle"] == "Reset":
                # TIME SERIES AUTO CAMERA: Mouse_Wheel: scale top/bottom equally, left/right = fixed
                if str(new_data_dict["toggle_view"]) in ["EEG_Time_Series", "ACCEL_Time_Series", "GYRO_Time_Series", "PPG_Time_Series"]:
                    rect = Rect(self.view[0].camera.rect)
                    line_data_buffer_swap = np.swapaxes(np.swapaxes(line_data_buffer, 0, 2)[1], 0, 1)        # FULL DATA
                    rect_x_scale = rect.left / len(line_data_buffer_swap[0])                # TimeSeries/BandChan
                    window_cutoff = int(rect_x_scale * len(line_data_buffer_swap[0]))
                    line_data_buffer_plot = line_data_buffer_swap[:, window_cutoff:]   # TimeSeries/BandChan
                    # update rect: Y LIMITS
                    line_data_y_limits = np.max(np.abs(line_data_buffer_plot), axis=1) + np.std(line_data_buffer_plot, axis=1)
                    for channel in range(0, len(line_data_buffer_swap)):
                        rect = Rect(self.view[channel].camera.rect)
                        rect.top, rect.bottom = np.max(line_data_y_limits), -np.max(line_data_y_limits)
                        self.view[channel].camera.rect = rect
                    self.cam_toggle = "Unlock"
                # AUTO CAMERA BANDCHAN: Mouse_Wheel: scale top, bottom = 0, left/right = fixed
                if str(new_data_dict["toggle_view"]) in ["EEG_TF_BandChan", "ACCEL_TF_BandChan", "GYRO_TF_BandChan", "PPG_TF_BandChan"]:
                    line_data_buffer_swap = np.swapaxes(np.swapaxes(line_data_buffer, 0, 2)[1], 0, 1)        # FULL DATA   
                    rect = Rect(self.view[0].camera.rect)
                    rect_x_scale = rect.left / len(line_data_buffer_swap[0])                # TimeSeries/BandChan
                    window_cutoff = int(rect_x_scale * len(line_data_buffer_swap[0]))
                    line_data_buffer_plot = line_data_buffer_swap[:, window_cutoff:]   # TimeSeries/BandChan
                    line_data_buffer_plot_bandchan = []
                    for channel in range(0, len(line_data_buffer_swap), len(band_range_buffer)):
                        line_data_buffer_plot_bandchan.append(line_data_buffer_plot[channel:channel + len(band_range_buffer)])
                    line_data_buffer_plot_bandchan = np.array(line_data_buffer_plot_bandchan)
                    for channel in range(0, n_chan_buffer):
                        rect = Rect(self.view[channel].camera.rect)
                        rect.top = np.max(line_data_buffer_plot_bandchan)
                        rect.bottom = 0
                        self.view[channel].camera.rect = rect
                    self.cam_toggle = "Unlock"
                # FFT/BAND AUTO CAMERA: Mouse_Wheel: scale top, bottom = 0, left/right = fixed
                if str(new_data_dict["toggle_view"]) in ["EEG_FFT", "ACCEL_FFT", "GYRO_FFT", "PPG_FFT", "EEG_TF_Band", "ACCEL_TF_Band", "GYRO_TF_Band", "PPG_TF_Band"]:
                    rect = Rect(self.view[0].camera.rect)
                    line_data_buffer_swap = np.swapaxes(np.swapaxes(line_data_buffer, 0, 2)[1], 0, 1)        # FULL DATA
                    if str(new_data_dict["toggle_view"]) in ["EEG_FFT", "ACCEL_FFT", "GYRO_FFT", "PPG_FFT"]:
                        rect_x_scale = rect.right / len(line_data_buffer_swap[0])                # FFT
                        window_cutoff = int(rect_x_scale * len(line_data_buffer_swap[0]))
                        line_data_buffer_plot = line_data_buffer_swap[:, 2:window_cutoff]        # FFT: NOTE FOR FFT we start at 2 to ignore low freq noise
                    else:
                        rect_x_scale = rect.left / len(line_data_buffer_swap[0])                 # TF_BAND
                        window_cutoff = int(rect_x_scale * len(line_data_buffer_swap[0]))
                        line_data_buffer_plot = line_data_buffer_swap[:, window_cutoff:len(line_data_buffer_swap[0])]        # TF_BAND
                    # update rect
                    rect = Rect(self.view[0].camera.rect)
                    rect.top = np.max(line_data_buffer_plot) + np.std(line_data_buffer_plot)
                    rect.bottom = 0
                    self.view[0].camera.rect = rect
                    self.cam_toggle = "Unlock"
            else:
                if str(new_data_dict["toggle_view"]) not in ["EEG_PSD", "ACCEL_PSD", "GYRO_PSD", "PPG_PSD"]:
                    self.cam_toggle = "Auto"

        # === AUTO CAMERA UPDATE
        if self.cam_toggle == "Auto":
            # TIME SERIES AUTO CAMERA: Mouse_Wheel: scale top/bottom equally, left/right = fixed
            if str(new_data_dict["toggle_view"]) in ["EEG_Time_Series", "ACCEL_Time_Series", "GYRO_Time_Series", "PPG_Time_Series"]:
                rect = Rect(self.view[0].camera.rect)
                line_data_buffer_swap = np.swapaxes(np.swapaxes(line_data_buffer, 0, 2)[1], 0, 1)        # FULL DATA
                rect_x_scale = rect.left / len(line_data_buffer_swap[0])                # TimeSeries/BandChan
                window_cutoff = int(rect_x_scale * len(line_data_buffer_swap[0]))
                line_data_buffer_plot = line_data_buffer_swap[:, window_cutoff:]   # TimeSeries/BandChan
                # update rect: Y LIMITS
                line_data_y_limits = np.max(np.abs(line_data_buffer_plot), axis=1) + np.std(line_data_buffer_plot, axis=1)
                for channel in range(0, len(line_data_buffer_swap)):
                    rect = Rect(self.view[channel].camera.rect)
                    rect.top, rect.bottom = line_data_y_limits[channel], -line_data_y_limits[channel]
                    self.view[channel].camera.rect = rect
            # FFT/BAND AUTO CAMERA: Mouse_Wheel: scale top, bottom = 0, left/right = fixed
            if str(new_data_dict["toggle_view"]) in ["EEG_FFT", "ACCEL_FFT", "GYRO_FFT", "PPG_FFT", "EEG_TF_Band", "ACCEL_TF_Band", "GYRO_TF_Band", "PPG_TF_Band"]:
                rect = Rect(self.view[0].camera.rect)
                line_data_buffer_swap = np.swapaxes(np.swapaxes(line_data_buffer, 0, 2)[1], 0, 1)        # FULL DATA
                if str(new_data_dict["toggle_view"]) in ["EEG_FFT", "ACCEL_FFT", "GYRO_FFT", "PPG_FFT"]:                    
                    rect_x_scale = rect.right / len(line_data_buffer_swap[0])                # FFT
                    window_cutoff = int(rect_x_scale * len(line_data_buffer_swap[0]))
                    line_data_buffer_plot = line_data_buffer_swap[:, 2:window_cutoff]        # FFT: NOTE FOR FFT we start at 2 to ignore low freq noise
                else:
                    rect_x_scale = rect.left / len(line_data_buffer_swap[0])                 # TF_BAND
                    window_cutoff = int(rect_x_scale * len(line_data_buffer_swap[0]))
                    line_data_buffer_plot = line_data_buffer_swap[:, window_cutoff:len(line_data_buffer_swap[0])]        # TF_BAND
                # update rect
                rect = Rect(self.view[0].camera.rect)
                rect.top = np.max(line_data_buffer_plot) + np.std(line_data_buffer_plot)
                rect.bottom = 0
                self.view[0].camera.rect = rect
            # AUTO CAMERA BANDCHAN: Mouse_Wheel: scale top, bottom = 0, left/right = fixed
            if str(new_data_dict["toggle_view"]) in ["EEG_TF_BandChan", "ACCEL_TF_BandChan", "GYRO_TF_BandChan", "PPG_TF_BandChan"]:
                line_data_buffer_swap = np.swapaxes(np.swapaxes(line_data_buffer, 0, 2)[1], 0, 1)        # FULL DATA   
                rect = Rect(self.view[0].camera.rect)
                rect_x_scale = rect.left / len(line_data_buffer_swap[0])                # TimeSeries/BandChan
                window_cutoff = int(rect_x_scale * len(line_data_buffer_swap[0]))
                line_data_buffer_plot = line_data_buffer_swap[:, window_cutoff:]   # TimeSeries/BandChan
                line_data_buffer_plot_bandchan = []
                for channel in range(0, len(line_data_buffer_swap), len(band_range_buffer)):
                    line_data_buffer_plot_bandchan.append(line_data_buffer_plot[channel:channel + len(band_range_buffer)])
                line_data_buffer_plot_bandchan = np.array(line_data_buffer_plot_bandchan)
                for channel in range(0, n_chan_buffer):
                    rect = Rect(self.view[channel].camera.rect)
                    rect.top = np.max(line_data_buffer_plot_bandchan[channel])
                    rect.bottom = 0
                    self.view[channel].camera.rect = rect

        # === AUTO CAMERA UPDATE PSD
        if self.current_cam_toggle_canvas != self.last_cam_toggle_canvas:
            if new_data_dict["cam_toggle"] == "Reset":
                # PSD AUTO/LOCK/UNLOCK MORLET SCALING: update cmap/clim for colormap/limits, PSD Z AXIS CAMERA: Mouse_Wheel: scale top/bottom equally, left/right = fixed
                if str(new_data_dict["toggle_view"]) in ["EEG_PSD", "ACCEL_PSD", "GYRO_PSD", "PPG_PSD"]:
                    rect = Rect(self.view[0].camera.rect)
                    rect_x_scale = rect.left / len(psd_data_buffer[0][0])
                    window_cutoff = int(rect_x_scale * len(psd_data_buffer[0][0]))
                    psd_data_buffer_plot = psd_data_buffer[:, :, window_cutoff:]
                    # PSD Unlock: Using FULL DATA WINDOW for scale
                    low_std_buffer, high_std_buffer = [], []
                    for channel in range(0, len(self.image)):
                        data_buffer = np.array(psd_data_buffer_plot[channel])
                        low_std_buffer.append(np.median(data_buffer) - np.std(data_buffer))
                        high_std_buffer.append(np.median(data_buffer) + (np.std(data_buffer) * 3))
                    for channel in range(0, len(self.image)):
                        clim_list = [0, 0]
                        clim_list[0], clim_list[1] = np.min(np.array(low_std_buffer)), np.max(np.array(high_std_buffer))
                        self.image[channel].clim = tuple(clim_list)
                    self.cam_toggle = "Unlock"
            else:
                self.cam_toggle = "Auto"
        else:
##        # PSD AUTO/LOCK/UNLOCK MORLET SCALING: update cmap/clim for colormap/limits, PSD Z AXIS CAMERA: Mouse_Wheel: scale top/bottom equally, left/right = fixed
            if str(new_data_dict["toggle_view"]) in ["EEG_PSD", "ACCEL_PSD", "GYRO_PSD", "PPG_PSD"]:
                rect = Rect(self.view[0].camera.rect)
                rect_x_scale = rect.left / len(psd_data_buffer[0][0])
                window_cutoff = int(rect_x_scale * len(psd_data_buffer[0][0]))
                psd_data_buffer_plot = psd_data_buffer[:, :, window_cutoff:]
                # PSD Unlock: Using FULL DATA WINDOW for scale
                if self.cam_toggle == "Auto":
                    for channel in range(0, len(self.image)):
                        clim_list = [0, 0]
                        data_buffer = np.array(psd_data_buffer_plot[channel])
                        clim_list[0] = np.median(data_buffer) - np.std(data_buffer)
                        clim_list[1] = np.median(data_buffer) + (np.std(data_buffer) * 3)
                        self.image[channel].clim = tuple(clim_list)
                # PSD Unlock: Using FULL DATA WINDOW for scale: MULTIPLY BY SELF.PSD_SCALE FOR UNLOCK
                if self.cam_toggle == "Unlock":
                    for channel in range(0, len(self.image)):
                        clim_list = [0, 0]
                        data_buffer = np.array(psd_data_buffer_plot[channel])
                        clim_list[0] = np.median(data_buffer) - ((np.std(data_buffer) * self.psd_scale))
                        clim_list[1] = np.median(data_buffer) + ((np.std(data_buffer) * 3) * self.psd_scale)
                        self.image[channel].clim = tuple(clim_list)

        # update last/current cam_toggle_canvas
        self.last_cam_toggle_canvas = self.current_cam_toggle_canvas

# ==================== REFACTOR CAMERA ====================


##            # === update SELF.COLORBAR PSD VIEW WORKING BASIC
##            for channel in range(0, len(self.image)):
####                self.colorbar.clim = self.image[channel].clim
####                self.view_colorbar.clim = self.image[channel].clim    
##                self.psd_colorbar_all[channel].clim = self.image[channel].clim
##                print('self.psd_colorbar_all[channel]')
##                for property, value in vars(self.psd_colorbar_all[channel]).items():
##                    print(property, ":", value)
##                print(correct)


####    # OPTIONAL: CONNECT from MyMainWindow pyqt to Canvas, requires: win.new_data_filt.connect(canvas_wrapper.update_toggles)
####    def update_toggles(self, new_data_dict_line_toggles):
####        print('TEST current_line_toggles: ' + str(self.line_bool_test))
####        print('TEST new_line_toggles:     ' + str(new_data_dict_line_toggles["line_toggles"]))
####        if new_data_dict_line_toggles["line_toggles"] != self.line_bool_test:
####            print('current_line_toggles: ' + str(self.line_bool_test))
####            print('new_line_toggles:     ' + str(new_data_dict_line_toggles["line_toggles"]))
####            self.line_bool_test = new_data_dict_line_toggles["line_toggles"]

##    # OPTIONAL: CONNECT from MyMainWindow pyqt to Canvas, requires: win.new_data_filt.connect(canvas_wrapper.update_toggles)
##    def update_toggles(self, new_data_dict_filt):
##        if str(new_data_dict_filt["view_toggle"]) != str(self.toggle_view):
##            print('current_view: ' + str(self.toggle_view))
##            print('new_view: ' + str(new_data_dict_filt["view_toggle"]))
##        self.toggle_view = new_data_dict_filt["view_toggle"]

class MyMainWindow(QtWidgets.QMainWindow):
    closing = QtCore.pyqtSignal()
    new_data_toggle = QtCore.pyqtSignal(dict)

    def __init__(self, canvas_wrapper: CanvasWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # === Define vars
        self.cam_toggle_count = 0
        self.play_bool, self.corner_widget_bool = True, False

        self.cam_toggle = "Auto"
        self.current_device_view = "EEG_Time_Series"
        self.view_device_bool, self.device_bool, self.view_bool = "EEG_Time_Series", "EEG", "Time_Series"
        self.artifact_bool, self.artifact_eeg, self.artifact_accel, self.artifact_gyro = "off", "off_eeg", "off_accel", "off_gyro"
        self.filter_bool, self.filter_eeg, self.filter_accel, self.filter_gyro, self.filter_ppg = "off", "off", "off_accel", "off_gyro", "off_ppg"

        # === QtWidgets: centralwidget/main_layout/menubar
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QGridLayout()
##        main_layout = QtWidgets.QHBoxLayout()
        self.menubar_master = QtWidgets.QMenuBar()

        # File Menu
        actionFile = self.menubar_master.addMenu("File")
        actionFile.addAction("Play")
        actionFile.addAction("Pause")
        actionFile.addAction("Exit")

        # SUBMENU CAMERA
        actionCam = self.menubar_master.addMenu("Camera")
        actionCam.addAction("Auto")
        actionCam.addAction("Reset")

        # === REFACTOR Device
        self.actionDevice = self.menubar_master.addMenu("Device")
        self.actionDevice.addAction("EEG")
        self.actionDevice.addAction("ACCEL")
        self.actionDevice.addAction("GYRO")
        self.actionDevice.addAction("PPG")

        # === REFACTOR View
        self.actionView = self.menubar_master.addMenu("View")
        self.actionView.addAction("Time_Series")
        self.actionView.addAction("FFT")
        self.actionView.addAction("TF_Band")
        self.actionView.addAction("TF_BandChan")        
        self.actionView.addAction("PSD")

        # === REFACTOR Filter
        self.all_filter_labels = {
                     "EEG":    ["off", "default", "delta/theta", "alpha", "beta", "gamma"],
                     "ACCEL":  ["off", "default", "low", "mid", "high", "highest"],
                     "GYRO":   ["off", "default", "low", "mid", "high", "highest"],
                     "PPG":    ["off", "default", "low", "mid", "high", "highest"]}
        self.actionFilter = self.menubar_master.addMenu("Filter")
        self.current_filter_labels = self.all_filter_labels["EEG"]
        for filt in range(0, len(self.current_filter_labels)):
            self.actionFilter.addAction(str(self.current_filter_labels[filt]))

        # connect controls
        actionFile.triggered[QAction].connect(self._connect_controls)
        actionCam.triggered[QAction].connect(self._connect_controls)
        self.actionDevice.triggered[QAction].connect(self._connect_controls)
        self.actionView.triggered[QAction].connect(self._connect_controls)
        self.actionFilter.triggered[QAction].connect(self._connect_controls)

##        # === QLABEL Device/View/Filter
##        self.current_view = QtWidgets.QLabel("Current_View: " + str(self.view_bool))
####        main_layout.addWidget(self.current_view)
##        main_layout.addWidget(self.current_view, 0, 1)

        # define current view/filter/camera QLabel
        self.display_info = QtWidgets.QGridLayout()
        self.current_device = QtWidgets.QLabel("Device: " + str(self.device_bool))
        self.display_info.addWidget(self.current_device, 0, 0, Qt.AlignVCenter)
        self.current_view = QtWidgets.QLabel("View: " + str(self.view_bool))
        self.display_info.addWidget(self.current_view, 0, 1, Qt.AlignVCenter)
        self.current_filter = QtWidgets.QLabel("Filter: " + str(self.filter_bool))
        self.display_info.addWidget(self.current_filter, 0, 2, Qt.AlignVCenter)

        # === RESIZE set stretch to keep top/bottom menu bars during resize
        self.display_info.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 2)

        # === QLABEL CornerWidget all device/views
        self.cornerWidget_dict = {
                     "EEG_Time_Series": "Detect Artifacts", "EEG_FFT": "Detect Artifacts",
                     "EEG_PSD": "Interpolate", "EEG_TF_Band": "Interpolate", "EEG_TF_BandChan": "Interpolate",
##                     "EEG_PSD": "Interpolate Artifacts", "EEG_TF_Band": "Interpolate Artifacts", "EEG_TF_BandChan": "Interpolate Artifacts",
                     "ACCEL_Time_Series": "Detect Movement", "ACCEL_FFT": "Detect Movement", "ACCEL_TF_Band": "Detect Movement", "ACCEL_TF_BandChan": "Detect Movement", "ACCEL_PSD": "Detect Movement",
                     "GYRO_Time_Series": "Detect Movement", "GYRO_FFT": "Detect Movement", "GYRO_TF_Band": "Detect Movement", "GYRO_TF_BandChan": "Detect Movement", "GYRO_PSD": "Detect Movement",
                     "PPG_Time_Series": "Detect Heart Rate", "PPG_FFT": "Detect Heart Rate", "PPG_TF_Band": "Detect Heart Rate", "PPG_TF_BandChan": "Detect Heart Rate", "PPG_PSD": "Detect Heart Rate"};
##        corner_widget_text = self.cornerWidget_dict["EEG_Time_Series"]
        self.corner_widget_text = self.cornerWidget_dict["EEG_Time_Series"]
        self.corner_widget = QtWidgets.QPushButton("      " + str(self.corner_widget_text) + " (off)      ")
        self.corner_widget.setStyleSheet('QPushButton {color: grey}')
        self.menubar_master.setCornerWidget(self.corner_widget)
        self.corner_widget.clicked.connect(lambda: self._connect_button_click_corner_widget(self.corner_widget))

        # === QPushButton bool for line on/off
        device_view_dict = {"EEG_Time_Series": ch_names_eeg, "EEG_FFT": ch_names_eeg,
                            "ACCEL_Time_Series": ch_names_accel, "GYRO_Time_Series": ch_names_gyro, "PPG_Time_Series": ch_names_ppg, "ACCEL_FFT": ch_names_accel, "GYRO_FFT": ch_names_gyro, "PPG_FFT": ch_names_ppg, 
                            "EEG_TF_BandChan": ['delta', 'theta', 'alpha', 'beta', 'gamma'], "EEG_TF_Band": ['delta', 'theta', 'alpha', 'beta', 'gamma'],
                            "ACCEL_TF_BandChan": ['broadband', 'low', 'mid', 'high', 'highest'], "GYRO_TF_BandChan": ['broadband', 'low', 'mid', 'high', 'highest'], "PPG_TF_BandChan": ['broadband', 'low', 'mid', 'high', 'highest'], 
                            "ACCEL_TF_Band": ['broadband', 'low', 'mid', 'high', 'highest'], "GYRO_TF_Band": ['broadband', 'low', 'mid', 'high', 'highest'], "PPG_TF_Band": ['broadband', 'low', 'mid', 'high', 'highest'], 
                            "EEG_PSD": ch_names_eeg, "ACCEL_PSD": ch_names_accel, "GYRO_PSD": ch_names_gyro, "PPG_PSD": ch_names_ppg}
        self.current_chan_labels = device_view_dict["EEG_Time_Series"]
        self.line_bool_win = [True]*len(self.current_chan_labels)
        self.button_info = QtWidgets.QGridLayout()
        self.button_text = self.current_chan_labels
        self.button_color = ['black', 'red', 'blue', 'green', 'purple']
        main_layout.addLayout(self.button_info, 3, 1)


####        # === optional: alternative BPM in its own box
####        self.current_bpm = 0
####        self.heart_bpm = QtWidgets.QLabel("BPM: " + str(self.current_bpm))
####        main_layout.addWidget(self.heart_bpm)

        # === Finish up
        main_layout.addWidget(self.menubar_master, 0, 1)
        self._canvas_wrapper = canvas_wrapper
        main_layout.addWidget(self._canvas_wrapper.canvas.native, 1, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # === Update QPushButton Labels and Buttons
    def _connect_button_click(self, button):
        text_buffer = str(button.text())
        device_view_dict_click = {"EEG_Time_Series": ch_names_eeg, "EEG_FFT": ch_names_eeg,
                            "ACCEL_Time_Series": ch_names_accel, "GYRO_Time_Series": ch_names_gyro, "PPG_Time_Series": ch_names_ppg, "ACCEL_FFT": ch_names_accel, "GYRO_FFT": ch_names_gyro, "PPG_FFT": ch_names_ppg, 
                            "EEG_TF_BandChan": ['delta', 'theta', 'alpha', 'beta', 'gamma'], "EEG_TF_Band": ['delta', 'theta', 'alpha', 'beta', 'gamma'],
                            "ACCEL_TF_BandChan": ['broadband', 'low', 'mid', 'high', 'highest'], "GYRO_TF_BandChan": ['broadband', 'low', 'mid', 'high', 'highest'], "PPG_TF_BandChan": ['broadband', 'low', 'mid', 'high', 'highest'], 
                            "ACCEL_TF_Band": ['broadband', 'low', 'mid', 'high', 'highest'], "GYRO_TF_Band": ['broadband', 'low', 'mid', 'high', 'highest'], "PPG_TF_Band": ['broadband', 'low', 'mid', 'high', 'highest'],
                            "EEG_PSD": ch_names_eeg, "ACCEL_PSD": ch_names_accel, "GYRO_PSD": ch_names_gyro, "PPG_PSD": ch_names_ppg}
        current_device_view_buffer = str(self.device_bool) + '_' + str(self.view_bool)
        self.line_labels_list = device_view_dict_click[str(current_device_view_buffer)]
        # create new list with " (off)" appended
        line_labels_off_list = []
        for label in range(0, len(self.line_labels_list)):
            line_labels_off_list.append(str(self.line_labels_list[label] + ' (off)'))
        if text_buffer in self.line_labels_list or text_buffer in line_labels_off_list:
            if text_buffer in self.line_labels_list:
                this_index = self.line_labels_list.index(text_buffer)
            else:
                this_index = line_labels_off_list.index(text_buffer)
            self.line_bool_win[this_index] = not self.line_bool_win[this_index]
            # === Change color
            if self.line_bool_win[this_index] == False:
                self.button_widgets[this_index].setText(str(self.line_labels_list[this_index] + ' (off)'))
##                self.button_widgets[this_index].setStyleSheet('QPushButton {color: white}')
                self.button_widgets[this_index].setStyleSheet('QPushButton {color: grey}')
            else:
                self.button_widgets[this_index].setText(str(self.line_labels_list[this_index]))
                self.button_widgets[this_index].setStyleSheet('QPushButton {color: ' + str(self.button_color[this_index]) + '}')

    # === CORNER WIDGET QPushButton UPDATE
    def _connect_button_click_corner_widget(self, button):
        text_buffer = str(button.text())
        self.cornerWidget_dict = {
                     "EEG_Time_Series": "Detect Artifacts", "EEG_FFT": "Detect Artifacts",
                     "EEG_PSD": "Interpolate", "EEG_TF_Band": "Interpolate", "EEG_TF_BandChan": "Interpolate",
##                     "EEG_PSD": "Interpolate Artifacts", "EEG_TF_Band": "Interpolate Artifacts", "EEG_TF_BandChan": "Interpolate Artifacts",
                     "ACCEL_Time_Series": "Detect Movement", "ACCEL_FFT": "Detect Movement", "ACCEL_TF_Band": "Detect Movement", "ACCEL_TF_BandChan": "Detect Movement", "ACCEL_PSD": "Detect Movement",
                     "GYRO_Time_Series": "Detect Movement", "GYRO_FFT": "Detect Movement", "GYRO_TF_Band": "Detect Movement", "GYRO_TF_BandChan": "Detect Movement", "GYRO_PSD": "Detect Movement",
                     "PPG_Time_Series": "Detect Heart Rate", "PPG_FFT": "Detect Heart Rate", "PPG_TF_Band": "Detect Heart Rate", "PPG_TF_BandChan": "Detect Heart Rate", "PPG_PSD": "Detect Heart Rate"};
        if self.corner_widget_bool == False:
            self.corner_widget_text = self.cornerWidget_dict[str(self.current_device_view)]
            self.corner_widget.setText('      ' + str(self.corner_widget_text) + '      ')
            self.corner_widget.setStyleSheet('QPushButton {color: black}')
        else:
            self.corner_widget_text = self.cornerWidget_dict[str(self.current_device_view)]
            self.corner_widget.setText('      ' + str(self.corner_widget_text) + ' (off)      ')
            self.corner_widget.setStyleSheet('QPushButton {color: grey}')
        # update corner_widget_bool
        self.corner_widget_bool = not self.corner_widget_bool

        # emit data
        data_dict_toggle = {"view_toggle": self.view_device_bool, "cam_toggle": self.cam_toggle, "play_bool": self.play_bool, "corner_widget_bool": self.corner_widget_bool, "cam_toggle_count": self.cam_toggle_count,
                            "filt_toggle_eeg": self.filter_eeg, "filt_toggle_accel": self.filter_accel, "filt_toggle_gyro": self.filter_gyro, "filt_toggle_ppg": self.filter_ppg, "line_toggles": self.line_bool_win}
        self.new_data_toggle.emit(data_dict_toggle)

    # update corner_widget for all device_views
    def update_corner_widget(self, new_corner_widget_dict):
        if self.corner_widget_bool == True:
            if self.current_device_view in ["EEG_Time_Series", "EEG_FFT"]:
                total_artifacts_eeg_test = new_corner_widget_dict["eeg_artifact"]
                self.corner_widget.setText("Detect Artifacts: " + str(total_artifacts_eeg_test))
            if self.current_device_view in ["EEG_PSD", "EEG_TF_Band", "EEG_TF_BandChan"]:
                total_artifacts_eeg_test = new_corner_widget_dict["eeg_artifact"]
                self.corner_widget.setText("Interpolate: " + str(total_artifacts_eeg_test))
##                self.corner_widget.setText("Interpolate Artifacts: " + str(total_artifacts_eeg_test))
            if self.current_device_view in ["ACCEL_Time_Series", "ACCEL_FFT", "ACCEL_TF_Band", "ACCEL_TF_BandChan", "ACCEL_PSD"]:
                total_artifacts_accel_test = new_corner_widget_dict["accel_movement"]
                self.corner_widget.setText("Detect Movement: " + str(total_artifacts_accel_test))
            if self.current_device_view in ["GYRO_Time_Series", "GYRO_FFT", "GYRO_TF_Band", "GYRO_TF_BandChan", "GYRO_PSD"]:
                total_artifacts_gyro_test = new_corner_widget_dict["gyro_movement"]
                self.corner_widget.setText("Detect Movement: " + str(total_artifacts_gyro_test))
            if self.current_device_view in ["PPG_Time_Series", "PPG_FFT", "PPG_TF_Band", "PPG_TF_BandChan", "PPG_PSD"]:
                self.current_bpm = new_corner_widget_dict["heart_bpm"]
                self.corner_widget.setText("Detect Heart Rate: " + str(self.current_bpm))

    def _connect_controls(self,q):
        text_buffer = q.text()

        # === Play/Pause/Exit commands
        if text_buffer in ["Play", "Pause", "Exit"]:
            if text_buffer == "Exit":
                exit()
            elif text_buffer == "Play":
                if self.play_bool == False:
                    self.play_bool = True
            elif text_buffer == "Pause":
                if self.play_bool == True:
                    self.play_bool = False

        # === Update last_device_view and self.current_device_view
        last_device_view = self.current_device_view
        if text_buffer in ["EEG", "ACCEL", "GYRO", "PPG"]:
            if self.device_bool != text_buffer:
                self.current_device_view = str(text_buffer) + '_' + str(self.view_bool)
        if text_buffer in ["Time_Series", "FFT", "TF_Band", "TF_BandChan", "PSD"]:
            if self.view_bool != text_buffer:
                self.current_device_view = str(self.device_bool) + '_' + str(text_buffer)





        # === QPushButton Update if view and/or device change
        device_view_dict = {"EEG_Time_Series": ch_names_eeg, "EEG_FFT": ch_names_eeg,
                            "ACCEL_Time_Series": ch_names_accel, "GYRO_Time_Series": ch_names_gyro, "PPG_Time_Series": ch_names_ppg, "ACCEL_FFT": ch_names_accel, "GYRO_FFT": ch_names_gyro, "PPG_FFT": ch_names_ppg, 
                            "EEG_TF_BandChan": ['delta', 'theta', 'alpha', 'beta', 'gamma'], "EEG_TF_Band": ['delta', 'theta', 'alpha', 'beta', 'gamma'],
                            "ACCEL_TF_BandChan": ['broadband', 'low', 'mid', 'high', 'highest'], "GYRO_TF_BandChan": ['broadband', 'low', 'mid', 'high', 'highest'], "PPG_TF_BandChan": ['broadband', 'low', 'mid', 'high', 'highest'], 
                            "ACCEL_TF_Band": ['broadband', 'low', 'mid', 'high', 'highest'], "GYRO_TF_Band": ['broadband', 'low', 'mid', 'high', 'highest'], "PPG_TF_Band": ['broadband', 'low', 'mid', 'high', 'highest'], 
                            "EEG_PSD": ch_names_eeg, "ACCEL_PSD": ch_names_accel, "GYRO_PSD": ch_names_gyro, "PPG_PSD": ch_names_ppg}
        if last_device_view != self.current_device_view:
            print("OLD -> NEW: self.current_device_view: " + str(last_device_view) + ', ' + str(self.current_device_view))
            # always remove QPushButtons because we reset the channel/band lines to all display, if range(self.button_info.count()) = 0 that's ok
            for i in reversed(range(self.button_info.count())):
                self.button_info.itemAt(i).widget().setParent(None)
            # QPushButton: channel toggle on/off lines
            self.button_widgets = []
##            self.button_text = self.button_channel_list
            self.button_text = device_view_dict[str(self.current_device_view)]
            self.button_color = ['black', 'red', 'blue', 'green', 'purple']
            for channel in range(0, len(self.button_text)):
                self.button_widgets.append(QtWidgets.QPushButton(str(self.button_text[channel])))
                self.button_widgets[channel].setStyleSheet('QPushButton {color: ' + str(self.button_color[channel]) + '}')
                self.button_info.addWidget(self.button_widgets[channel], 0, channel, Qt.AlignVCenter)
            # === QPushButton: connect clicks
            # ACCEL/GYRO/PPG_Time_Series/FFT: 3 channels
            self.button_widgets[0].clicked.connect(lambda: self._connect_button_click(self.button_widgets[0]))
            self.button_widgets[1].clicked.connect(lambda: self._connect_button_click(self.button_widgets[1]))
            self.button_widgets[2].clicked.connect(lambda: self._connect_button_click(self.button_widgets[2]))
            # ACCEL/GYRO/PPG_Time_Series/FFT: 3 channels
            if len(self.button_text) == 4:
                self.button_widgets[3].clicked.connect(lambda: self._connect_button_click(self.button_widgets[3]))
            # BAND/BAND_CHAN: 5 BANDS: EEG_TF_BandChan: 4 channels, ACCEL/GYRO/PPG_TF_BandChan: 3 channels, EEG/ACCEL/GYRO/PPG_TF_Band: 1 channels
            if len(self.button_text) == 5:
                self.button_widgets[3].clicked.connect(lambda: self._connect_button_click(self.button_widgets[3]))
                self.button_widgets[4].clicked.connect(lambda: self._connect_button_click(self.button_widgets[4]))
            # === QPushButton bool for line on/off
            self.line_bool_win = [True]*len(self.button_text)





        # REFACTOR VIEW/DEVICE SEPARATE
        if text_buffer in ["EEG", "ACCEL", "GYRO", "PPG"]:
            current_device_view = str(text_buffer) + '_' + str(self.view_bool)
            # === REFACTOR: update DEVICE change
            if self.device_bool != text_buffer:
                # === update Filter menu items for current view
                self.actionFilter.clear()
                for filt in range(0, len(self.all_filter_labels[text_buffer])):
                    self.actionFilter.addAction(str(self.all_filter_labels[text_buffer][filt]))
                self.current_filter_labels = self.all_filter_labels[text_buffer]
            # update current device
            self.device_bool = text_buffer
            self.current_device.setText("Current_Device: " + str(self.device_bool))
            # update current filter
            if self.device_bool == "EEG":
                self.filter_bool = self.filter_eeg
            if self.device_bool == "ACCEL":
                self.filter_bool = self.filter_accel
            if self.device_bool == "GYRO":
                self.filter_bool = self.filter_gyro
            if self.device_bool == "PPG":
                self.filter_bool = self.filter_ppg
            # update filter text and self.view_device_bool
            self.current_filter.setText("Current_Filter: " + str(self.filter_bool))
            self.view_device_bool = current_device_view
        # update current view
        if text_buffer in ["Time_Series", "FFT", "TF_Band", "TF_BandChan", "PSD"]:
            current_device_view = str(self.device_bool) + '_' + str(text_buffer)
            self.view_bool = text_buffer
            self.current_view.setText("Current_View: " + str(self.view_bool))
            self.view_device_bool = current_device_view






        # === REFACTOR Filter
        if text_buffer in ['off', 'default', 'delta/theta', 'alpha', 'beta', 'gamma', 'low', 'mid', 'high', 'highest']:
            if self.device_bool == "EEG":
                self.filter_eeg = text_buffer
                self.filter_bool = text_buffer
            if self.device_bool == "ACCEL":
                self.filter_accel = text_buffer + "_accel"
                self.filter_bool = text_buffer
            if self.device_bool == "GYRO":
                self.filter_gyro = text_buffer + "_gyro"
                self.filter_bool = text_buffer
            if self.device_bool == "PPG":
                self.filter_ppg = text_buffer + "_ppg"
                self.filter_bool = text_buffer
            self.current_filter.setText("Current_Filter: " + str(self.filter_bool))

        # === CORNER WIDGET QPushButton UPDATE
        self.cornerWidget_dict = {
                     "EEG_Time_Series": "Detect Artifacts", "EEG_FFT": "Detect Artifacts",
                     "EEG_PSD": "Interpolate Artifacts", "EEG_TF_Band": "Interpolate Artifacts", "EEG_TF_BandChan": "Interpolate Artifacts",
                     "ACCEL_Time_Series": "Detect Movement", "ACCEL_FFT": "Detect Movement", "ACCEL_TF_Band": "Detect Movement", "ACCEL_TF_BandChan": "Detect Movement", "ACCEL_PSD": "Detect Movement",
                     "GYRO_Time_Series": "Detect Movement", "GYRO_FFT": "Detect Movement", "GYRO_TF_Band": "Detect Movement", "GYRO_TF_BandChan": "Detect Movement", "GYRO_PSD": "Detect Movement",
                     "PPG_Time_Series": "Detect Heart Rate", "PPG_FFT": "Detect Heart Rate", "PPG_TF_Band": "Detect Heart Rate", "PPG_TF_BandChan": "Detect Heart Rate", "PPG_PSD": "Detect Heart Rate"};
        if last_device_view != self.current_device_view:
            if text_buffer in ["EEG", "ACCEL", "GYRO", "PPG"]:
                self.corner_widget_text = self.cornerWidget_dict[str(self.current_device_view)]
                self.corner_widget.setText('      '  + str(self.corner_widget_text) + ' (off)      ')
                self.corner_widget.setStyleSheet('QPushButton {color: grey}')
                self.corner_widget_bool = False
            else:
                if self.current_device_view in ["EEG_Time_Series", "EEG_FFT", "EEG_TF_Band", "EEG_TF_BandChan", "EEG_PSD"]:
                    if last_device_view in ["EEG_Time_Series", "EEG_FFT"]:
                        if self.current_device_view in ["EEG_TF_Band", "EEG_TF_BandChan", "EEG_PSD"]:
                            self.corner_widget_text = self.cornerWidget_dict[str(self.current_device_view)]
                            self.corner_widget.setText('      ' + str(self.corner_widget_text) + ' (off)      ')
                            self.corner_widget.setStyleSheet('QPushButton {color: grey}')
                            self.corner_widget_bool = False
                    if last_device_view in ["EEG_TF_Band", "EEG_TF_BandChan", "EEG_PSD"]:
                        if self.current_device_view in ["EEG_Time_Series", "EEG_FFT"]:
                            self.corner_widget_text = self.cornerWidget_dict[str(self.current_device_view)]
                            self.corner_widget.setText('      ' + str(self.corner_widget_text) + '      ')
                            self.corner_widget.setStyleSheet('QPushButton {color: grey}')            
                            self.corner_widget_bool = False

        # === Camera
        if text_buffer in ['Auto', 'Reset']:
            self.cam_toggle_count += 1
            self.cam_toggle = text_buffer

        # emit data
        data_dict_toggle = {"view_toggle": self.view_device_bool, "cam_toggle": self.cam_toggle, "play_bool": self.play_bool, "corner_widget_bool": self.corner_widget_bool, "cam_toggle_count": self.cam_toggle_count,
                            "filt_toggle_eeg": self.filter_eeg, "filt_toggle_accel": self.filter_accel, "filt_toggle_gyro": self.filter_gyro, "filt_toggle_ppg": self.filter_ppg, "line_toggles": self.line_bool_win}
        self.new_data_toggle.emit(data_dict_toggle)


    def closeEvent(self, event):
        print("Closing main window!")
        self.closing.emit()
        return super().closeEvent(event)


class DataSource(QtCore.QObject):
    """Object representing a complex data producer."""
    new_data = QtCore.pyqtSignal(dict)
    new_data_corner_widget = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()

    def __init__(self, num_iterations=100000, parent=None):
        super().__init__(parent)

        # original vars
        self._should_end = False
        self._num_iters = num_iterations

        self.last_samples_eeg = np.zeros(12)

        # define vars
        self.cam_toggle_count = 0
        
        self.toggle_catch_up = False
        self.play_bool, self.corner_widget_bool = True, False
        self.max_samples_eeg, self.max_samples_accel, self.max_samples_gyro, self.max_samples_ppg = int(sampling_rate_eeg * 3), int(sampling_rate_accel * 3), int(sampling_rate_gyro * 3), int(sampling_rate_ppg * 3)

        # === BACKEND TESTING
        self.current_time_buffer = local_clock()
        self.artifact_detect_lsl, self.artifact_detect_lsl_accel, self.artifact_detect_lsl_gyro = 0, 0, 0

        # define global vars
        self.af = [1.0]
        self.fft_bpm_ppg = 0
        self.last_time_test = local_clock()
        self.line_bool_data_source = [True]*n_chan_eeg
        self.cam_toggle, self.toggle_device_view = "Auto", "EEG_Time_Series"
        self.toggle_filt_eeg, self.toggle_filt_accel, self.toggle_filt_gyro, self.toggle_filt_ppg = "off", "off_accel", "off_gyro", "off_ppg"

        # === Time_Series: line/data vars
        self.data_eeg, self.lsl_eeg, self._line_data_eeg = np.zeros((window_eeg, n_chan_eeg)), np.zeros(window_eeg), create_line_variables(window_eeg, n_chan_eeg)
        self.data_accel, self.lsl_accel, self._line_data_accel = np.zeros((window_accel, n_chan_accel)), np.zeros(window_accel), create_line_variables(window_accel, n_chan_accel)
        self.data_gyro, self.lsl_gyro, self._line_data_gyro = np.zeros((window_gyro, n_chan_gyro)), np.zeros(window_gyro), create_line_variables(window_gyro, n_chan_gyro)
        self.data_ppg, self.lsl_ppg, self._line_data_ppg = np.zeros((window_ppg, n_chan_ppg)), np.zeros(window_ppg), create_line_variables(window_ppg, n_chan_ppg)
        # === Time_Series: FILTER vars
        self.eeg_filter_list_range, self.accel_filter_list_range, self.gyro_filter_list_range, self.ppg_filter_list_range = eeg_filter_list_final, accel_filter_list_final, gyro_filter_list_final, ppg_filter_list_final
        self.filter_list_text_eeg, self.filter_list_text_accel, self.filter_list_text_gyro, self.filter_list_text_ppg = eeg_filter_list_text_final, accel_filter_list_text_final, gyro_filter_list_text_final, ppg_filter_list_text_final
        # === create_data_filter_variables
        self.current_filt, self.current_filt_accel, self.current_filt_gyro, self.current_filt_ppg = 0, 0, 0, 0
        self.data_eeg_filter_all, self.lsl_eeg_corrected, self.filt_time_index_corrected, self.bf_eeg_all, self.zi_eeg_all, self.filt_state_eeg_all = create_data_filter_variables(
            window_eeg, n_chan_eeg, sampling_rate_eeg, self.eeg_filter_list_range)
##        self.data_accel_filter_all, self.lsl_accel_corrected, self.filt_time_index_corrected, self.bf_accel_all, self.zi_accel_all, self.filt_state_accel_all = create_data_filter_variables(
##            window_accel, n_chan_accel, sampling_rate_accel, self.accel_filter_list_range)
        self.data_accel_filter_all, self.lsl_accel_corrected, self.filt_time_index_accel_corrected, self.bf_accel_all, self.zi_accel_all, self.filt_state_accel_all = create_data_filter_variables(
            window_accel, n_chan_accel, sampling_rate_accel, self.accel_filter_list_range)
        self.data_gyro_filter_all, self.lsl_gyro_corrected, self.filt_time_index_gyro_corrected, self.bf_gyro_all, self.zi_gyro_all, self.filt_state_gyro_all = create_data_filter_variables(
            window_gyro, n_chan_gyro, sampling_rate_gyro, self.gyro_filter_list_range)
        self.data_ppg_filter_all, self.lsl_ppg_corrected, self.filt_time_index_ppg_corrected, self.bf_ppg_all, self.zi_ppg_all, self.filt_state_ppg_all = create_data_filter_variables(
            window_ppg, n_chan_ppg, sampling_rate_ppg, self.ppg_filter_list_range)
        # === FFT: line/data vars
        self.data_eeg_fft, self._line_data_eeg_fft = np.zeros([window_eeg_fft, n_chan_eeg_fft]), create_line_variables(window_eeg_fft, n_chan_eeg_fft)
        self.data_accel_fft, self._line_data_accel_fft = np.zeros([window_accel_fft, n_chan_accel_fft]), create_line_variables(window_accel_fft, n_chan_accel_fft)
        self.data_gyro_fft, self._line_data_gyro_fft = np.zeros([window_gyro_fft, n_chan_gyro_fft]), create_line_variables(window_gyro_fft, n_chan_gyro_fft)
        self.data_ppg_fft, self._line_data_ppg_fft = np.zeros([window_ppg_fft, n_chan_ppg_fft]), create_line_variables(window_ppg_fft, n_chan_ppg_fft)
        # === TF: line/filter vars: single/multiplot
##        self.band_range_eeg, self.band_range_accel, self.band_range_gyro, self.band_range_ppg = eeg_filter_list_final, accel_filter_list_final, gyro_filter_list_final, ppg_filter_list_final
        self.band_range_eeg, self.band_range_accel, self.band_range_gyro, self.band_range_ppg = eeg_band_list_final, accel_band_list_final, gyro_band_list_final, ppg_band_list_final
        self._line_data_eeg_tf_multiplot, self._line_data_eeg_tf_singleplot = create_tf_line_variables(window_eeg_tf, n_chan_eeg_tf, self.eeg_filter_list_range, self.band_range_eeg)
        self._line_data_accel_tf_multiplot, self._line_data_accel_tf_singleplot = create_tf_line_variables(window_accel_tf, n_chan_accel_tf, self.accel_filter_list_range, self.band_range_accel)
        self._line_data_gyro_tf_multiplot, self._line_data_gyro_tf_singleplot = create_tf_line_variables(window_gyro_tf, n_chan_gyro_tf, self.gyro_filter_list_range, self.band_range_gyro)
        self._line_data_ppg_tf_multiplot, self._line_data_ppg_tf_singleplot = create_tf_line_variables(window_ppg_tf, n_chan_ppg_tf, self.ppg_filter_list_range, self.band_range_ppg)
        self.data_eeg_tf, self.data_eeg_tf_filter = np.zeros((len(self.band_range_eeg), window_eeg_tf, n_chan_eeg)), np.zeros((len(self.eeg_filter_list_range), len(self.band_range_eeg), window_eeg_tf, n_chan_eeg))
        self.data_accel_tf, self.data_accel_tf_filter = np.zeros((len(self.band_range_accel), window_accel_tf, n_chan_accel)), np.zeros((len(self.accel_filter_list_range), len(self.band_range_accel), window_accel_tf, n_chan_accel))
        self.data_gyro_tf, self.data_gyro_tf_filter = np.zeros((len(self.band_range_gyro), window_gyro_tf, n_chan_gyro)), np.zeros((len(self.gyro_filter_list_range), len(self.band_range_gyro), window_gyro_tf, n_chan_gyro))
        self.data_ppg_tf, self.data_ppg_tf_filter = np.zeros((len(self.band_range_ppg), window_ppg_tf, n_chan_ppg)), np.zeros((len(self.ppg_filter_list_range), len(self.band_range_ppg), window_ppg_tf, n_chan_ppg))
        # === PSD EEG/ACCEL/GYRO/PPG
        self.image_data_eeg, self.plot_data_psd_eeg, self.data_eeg_psd = np.zeros([n_chan_eeg, freqNum_psd_eeg, window_eeg]), np.zeros([n_chan_eeg, len(freqs_psd_eeg), window_eeg]), np.zeros((n_chan_eeg, len(freqs_psd_eeg), window_eeg))
        self.image_data_accel, self.plot_data_psd_accel, self.data_accel_psd = np.zeros([n_chan_accel, freqNum_psd_accel, window_accel]), np.zeros([n_chan_accel, len(freqs_psd_accel), window_accel]), np.zeros((n_chan_accel, len(freqs_psd_accel), window_accel))
        self.image_data_gyro, self.plot_data_psd_gyro, self.data_gyro_psd = np.zeros([n_chan_gyro, freqNum_psd_gyro, window_gyro]), np.zeros([n_chan_gyro, len(freqs_psd_gyro), window_gyro]), np.zeros((n_chan_gyro, len(freqs_psd_gyro), window_gyro))
        self.image_data_ppg, self.plot_data_psd_ppg, self.data_ppg_psd = np.zeros([n_chan_ppg, freqNum_psd_ppg, window_ppg]), np.zeros([n_chan_ppg, len(freqs_psd_ppg), window_ppg]), np.zeros((n_chan_ppg, len(freqs_psd_ppg), window_ppg))
        # ========= ARTIFACT EEG NEW =========
        # NOTE: self.bf_eeg_index_offset[filt] = int(len(self.bf_eeg_all[filt]) / 2)
        # self.bf_eeg_index_offset[n_filt + 1] with 0 offset for raw: self.bf_eeg_index_offset[0] = 0
        # === Time Series Artifact vars
        self.artifact_pos_all = [[], [], [], []]
        self.bf_eeg_index_offset = np.zeros(len(self.eeg_filter_list_range) + 1)
        for filt in range(0, len(self.eeg_filter_list_range)):
            filt_time_buffer = int(len(self.bf_eeg_all[filt]) / 2)# * (1 / sampling_rate_eeg)
            self.bf_eeg_index_offset[filt + 1] = filt_time_buffer
        # === PSD/TF Artifact vars
        self.full_artifact_mask = np.zeros((n_chan_eeg, int(window_eeg + np.max(self.bf_eeg_index_offset))))
        # ========= ARTIFACT EEG NEW =========

        # ========= ARTIFACT ACCEL/GYRO NEW =========
        self.artifact_pos_all_accel, self.artifact_pos_all_gyro = [[], [], []], [[], [], []]
        self.bf_accel_index_offset = np.zeros(len(self.accel_filter_list_range) + 1)
        self.bf_gyro_index_offset = np.zeros(len(self.gyro_filter_list_range) + 1)
        for filt in range(0, len(self.accel_filter_list_range)):
            filt_time_buffer = int(len(self.bf_accel_all[filt]) / 2)# * (1 / sampling_rate_eeg)
            self.bf_accel_index_offset[filt + 1] = filt_time_buffer
        for filt in range(0, len(self.gyro_filter_list_range)):
            filt_time_buffer = int(len(self.bf_gyro_all[filt]) / 2)# * (1 / sampling_rate_eeg)
            self.bf_gyro_index_offset[filt + 1] = filt_time_buffer
        self.full_artifact_mask_accel = np.zeros((n_chan_accel, int(window_accel + np.max(self.bf_accel_index_offset))))
        self.full_artifact_mask_gyro = np.zeros((n_chan_gyro, int(window_gyro + np.max(self.bf_gyro_index_offset))))
        # ========= ARTIFACT ACCEL/GYRO NEW =========

        # ========= ARTIFACT PPG JUST FOR BF FILT =========
        self.bf_ppg_index_offset = np.zeros(len(self.ppg_filter_list_range) + 1)
        for filt in range(0, len(self.ppg_filter_list_range)):
            filt_time_buffer = int(len(self.bf_ppg_all[filt]) / 2)# * (1 / sampling_rate_eeg)
            self.bf_ppg_index_offset[filt + 1] = filt_time_buffer
        # ========= ARTIFACT PPG JUST FOR BF FILT =========

        # track heart rate ppg in array for standard deviation test for messy data = ??
        self.ppg_heart_rate_bpm_tracker = np.zeros(int(sampling_rate_ppg / 4))
        # total sample tracking
        self.total_samples_eeg, self.total_samples_accel, self.total_samples_gyro, self.total_samples_ppg = 0, 0, 0, 0

    def run_data_creation(self):

        # === EEG TREE
        if self.toggle_device_view in ["EEG_Time_Series", "EEG_FFT", "EEG_TF_Band", "EEG_TF_BandChan", "EEG_PSD"]:
            if self.toggle_device_view in ["EEG_TF_Band", "EEG_TF_BandChan"]:
                samples_eeg, timestamps_eeg = inlet_eeg_tf.pull_chunk(timeout=0.0, max_samples=self.max_samples_eeg)
            else:
                samples_eeg, timestamps_eeg = inlet_eeg.pull_chunk(timeout=0.0, max_samples=self.max_samples_eeg)
##            samples_eeg, timestamps_eeg = inlet_eeg_tf.pull_chunk(timeout=0.0, max_samples=self.max_samples_eeg)
####            samples_eeg, timestamps_eeg = inlet_eeg.pull_chunk(timeout=0.0, max_samples=self.max_samples_eeg)
######            samples_eeg, timestamps_eeg = inlet_eeg.pull_chunk(timeout=0.0, max_samples=64)

            if timestamps_eeg:

                print_me = np.array(samples_eeg)[:, 0]
                if print_me in self.last_samples_eeg:
                    print('self.last_samples_eeg: ' + str(self.last_samples_eeg))
                    print('samples_eeg: ' + str(print_me))
##                    print('TRUE')
##                else:
##                    print('FALSE')

                self.last_samples_eeg = print_me

##                print_me = np.array(samples_eeg)[:, 0]
##                print('print_me: ' + str(print_me))
                

                
                # === update eeg lsl/samples
                self.lsl_eeg = np.roll(self.lsl_eeg, -len(timestamps_eeg))
                self.lsl_eeg[(len(self.lsl_eeg) - len(timestamps_eeg)):len(self.lsl_eeg)] = timestamps_eeg
    ##            # FOUR CHANNEL: simply remove channel 5, keep same format, samples_eeg[sample][channel] where channels are not reversed yet
    ##            if toggle_aux == False:
    ##                samples_eeg = np.swapaxes(np.swapaxes(samples_eeg, 0, 1)[:4], 0, 1)
                samples_eeg = np.swapaxes(np.swapaxes(samples_eeg, 0, 1)[:n_chan_eeg], 0, 1)
                samples_eeg = np.array(samples_eeg)[:, ::-1]                # reverses channels
                self.data_eeg = np.vstack([self.data_eeg, samples_eeg])     # adds newest samples to end of len(self.data_eeg)
                # NOTE: self.data_eeg format self.data_eeg[sample][channel] ALREADY HAS CHANNELS REVERSED
                self.data_eeg = self.data_eeg[-window_eeg:]              # removes oldest samples at start of self.data_eeg
                # === ARTIFACT update eeg artifact_mask tracking
##                self.artifact_pos = np.roll(self.artifact_pos, -len(timestamps_eeg), axis=1)
                self.full_artifact_mask = np.roll(self.full_artifact_mask, -len(timestamps_eeg), axis=1)
                self.full_artifact_mask[:, -len(timestamps_eeg):] = 0
                # === EEG Filtering: track each filtered data
                filt_samples_eeg_buffer = np.zeros((len(self.eeg_filter_list_range), len(samples_eeg), n_chan_eeg))
                for filt in range(0, len(self.eeg_filter_list_range)):
                    filt_samples_eeg_buffer[filt], self.filt_state_eeg_all[filt] = lfilter(self.bf_eeg_all[filt], self.af, samples_eeg, axis=0, zi=self.filt_state_eeg_all[filt])
                    stack_data_buffer = np.vstack([self.data_eeg_filter_all[filt], filt_samples_eeg_buffer[filt]])
                    self.data_eeg_filter_all[filt] = stack_data_buffer[-window_eeg:]
                # === EEG Filter Time Correct lsl time for filter time lag
                self.lsl_eeg_corrected = np.zeros([len(self.eeg_filter_list_range), len(self.lsl_eeg)])
                for filt in range(0, len(self.eeg_filter_list_range)):
                    filt_time_buffer = (len(self.bf_eeg_all[filt]) / 2) * (1 / sampling_rate_eeg)
                    self.lsl_eeg_corrected[filt] = self.lsl_eeg + filt_time_buffer
                # === EEG Filtering Set PLOT data eeg
##                self.current_filt = self.filter_list_text.index(self.toggle_filt_eeg)
                self.current_filt = self.filter_list_text_eeg.index(self.toggle_filt_eeg)
                if self.current_filt == 0:
                    self.plot_data_eeg = (self.data_eeg - self.data_eeg.mean(axis=0))
                else:
                    plot_data_eeg_buffer = self.data_eeg_filter_all[self.current_filt - 1]
                    self.plot_data_eeg = plot_data_eeg_buffer

##                # === control for filter edge artifacts at begining of file
####                print('self.bf_eeg_index_offset: ' + str(self.bf_eeg_index_offset))
##                self.total_samples_eeg += len(timestamps_eeg)
##                range_max_buffer_eeg = len(self.data_eeg) - self.total_samples_eeg + (np.max(self.bf_eeg_index_offset) * 2)
##                if self.current_filt != 0:
##                    if self.total_samples_eeg < len(self.data_eeg):
##                        if range_max_buffer_eeg <= len(self.data_eeg):
##                            self.plot_data_eeg[0:int(range_max_buffer_eeg), :] = 0
##                        else:
##                            self.plot_data_eeg[:, :] = 0

                # === ARTIFACT REJECTION from PetalDisplay_v9.py
                if self.corner_widget_bool == True:
                    # window = 1024 = 4 seconds: since max(len(self.bf_eeg_all)) = 422 and 512 window doesn't miss samples, artifact detect in last 512 + 512 = 4 seconds
                    # interval = 256 = 1 second: perform artifact detection every second for window = last 4 seconds of data
                    artifact_interval = 1                               # perform artifact rejection every artifact_interval seconds
                    artifact_window = int(sampling_rate_eeg * 4)        # artifact window = 1024 = 4 seconds
                    if (local_clock() - self.artifact_detect_lsl) > artifact_interval:
                        # define start/end sample index for artifact_data_eeg with last artifact_window samples of data
                        artifact_data_eeg_range = [len(self.data_eeg) - artifact_window, len(self.data_eeg)]
                        artifact_data_eeg = np.swapaxes(self.data_eeg[artifact_data_eeg_range[0]:artifact_data_eeg_range[1]], 0, 1)
    ##                    artifact_data_eeg = np.swapaxes(self.data_eeg[-artifact_win_buffer:, :], 0, 1)
                        artifact_time_eeg, artifact_mask_eeg = artifact_rejection_eeg(artifact_data_eeg, sampling_rate_eeg)
                        # get artifact_time_eeg_index for raw and artifact_time_eeg_index_corrected for filter time/index correction
                        artifact_time_eeg_index = get_artifact_time_index_raw(artifact_time_eeg, artifact_data_eeg_range)
                        artifact_time_eeg_index_corrected = get_artifact_time_index_corrected(artifact_time_eeg, artifact_data_eeg_range, self.bf_eeg_index_offset)
                        self.full_artifact_mask = update_full_artifact_mask(self.full_artifact_mask, artifact_time_eeg_index, self.bf_eeg_index_offset)
                        # === ARTIFACT LINES NEW
                        if len(artifact_time_eeg_index) > 0:
                            self.artifact_pos_all = artifact_time_eeg_index
                        else:
                            self.artifact_pos_all = [[], [], [], []]
                        # === reset artifact detection lsl clock
                        self.artifact_detect_lsl = local_clock()

                        # UPDATE CornerWidget EEG TOTAL ARTIFACT COUNT
##                        total_eeg_artifact_test = 0
##                        for channel in range(0, len(self.artifact_pos_all)):
##                            if len(self.artifact_pos_all[channel]) > 0:
##                                total_eeg_artifact_test += int(len(self.artifact_pos_all[channel]) / 2)
##    ##                    print('total_eeg_artifact_test: ' + str(total_eeg_artifact_test))
##    ##                    total_eeg_artifact_test = sum(self.artifact_pos_all)
##                        data_dict_corner_widget = {"heart_bpm": "None", "eeg_artifact": str(total_eeg_artifact_test), "accel_movement": "None", "gyro_movement": "None"}
##                        self.new_data_corner_widget.emit(data_dict_corner_widget)

                        # UPDATE CornerWidget EEG CHANNEL ARTIFACT NAMES
                        eeg_artifact_test = ""
                        for channel in range(0, len(self.artifact_pos_all)):
                            if len(self.artifact_pos_all[channel]) > 0:
                                if eeg_artifact_test == "":
                                    eeg_artifact_test = str(ch_names_eeg[channel])
                                else:
                                    eeg_artifact_test = eeg_artifact_test + ", " + str(ch_names_eeg[channel])
                        if eeg_artifact_test == "":
                            eeg_artifact_test = "None"
                        data_dict_corner_widget = {"heart_bpm": "None", "eeg_artifact": str(eeg_artifact_test), "accel_movement": "None", "gyro_movement": "None"}
                        self.new_data_corner_widget.emit(data_dict_corner_widget)

                # === UPDATE LINE EEG data
                self._line_data_eeg = self._update_line_data_eeg()

                # === EEG FFT
                if self.toggle_device_view == "EEG_FFT":
                    data_eeg_fft = []
                    input_data_eeg_fft = np.swapaxes(self.plot_data_eeg, 0, 1)
                    for channel in range(0, n_chan_eeg_fft):
                        fft_data_buffer, fft_freq_buffer = getFFT(input_data_eeg_fft[channel][-window_eeg_fft_trim:], sampling_rate_eeg)
                        data_eeg_fft.append(fft_data_buffer)
                    data_eeg_fft, freq_eeg_fft = np.asarray(data_eeg_fft), fft_freq_buffer
                    self.data_eeg_fft = data_eeg_fft
                    self.plot_data_eeg_fft = (np.swapaxes(self.data_eeg_fft, 0, 1))
##                    self.plot_data_eeg_fft = (np.swapaxes(self.data_eeg_fft, 0, 1)) * 100
                    self._line_data_eeg_fft = self._update_line_data_eeg_fft()

                # === PSD TIME FREQUENCY WITH ARTIFACT REJECT
                if self.toggle_device_view == "EEG_PSD":
                    if self.current_filt == 0:
                        tf_data_buffer = (self.data_eeg - self.data_eeg.mean(axis=0))
                        tf_data_buffer = np.array([np.swapaxes(tf_data_buffer, 0, 1)])
                    else:
                        tf_data_buffer = np.array([np.swapaxes(self.data_eeg_filter_all[self.current_filt - 1], 0, 1)])
                    morlet_psd = tfr_array_morlet(tf_data_buffer, sfreq=sampling_rate_eeg, freqs=freqs_psd_eeg, output='power')[0]
                    # ========= PLOT PSD NEW =========
                    start_buf = int(np.max(self.bf_eeg_index_offset) - self.bf_eeg_index_offset[self.current_filt])
                    end_buf = int((window_eeg + np.max(self.bf_eeg_index_offset)) - self.bf_eeg_index_offset[self.current_filt])
                    artifact_mask_buffer = self.full_artifact_mask[:, start_buf:end_buf]

                    # interpolate
                    if self.corner_widget_bool == True:
                        for channel in range(0, len(morlet_psd)):
                            mask_buffer = (artifact_mask_buffer[channel] == 1)
                            for this_freq in range(0, len(morlet_psd[channel])):
##                                if self.clean_toggle != "off":
                                morlet_psd[channel, this_freq, mask_buffer] = np.median(morlet_psd[channel][this_freq])

##                                # MEAN/MEDIAN/MASK INTERPOLATION OPTIONS
##                                morlet_psd[channel, this_freq, mask_buffer] = np.median(morlet_psd[channel][this_freq])
######                                if self.artifact_toggle == "artifact_mask":
######                                    morlet_psd[channel, this_freq, mask_buffer] = 1
######                                elif self.artifact_toggle == "artifact_mean":
######                                    morlet_psd[channel, this_freq, mask_buffer] = np.mean(morlet_psd[channel][this_freq])
######                                else:                   # elif self.artifact_toggle == "artifact_median":
######                                    morlet_psd[channel, this_freq, mask_buffer] = np.median(morlet_psd[channel][this_freq])
########                                    morlet_psd[channel, this_freq, mask_buffer] = np.median(morlet_psd[channel][this_freq] - np.mean(morlet_psd[channel][this_freq]))
########                                morlet_psd[channel, :, mask_buffer] = np.median(morlet_psd[channel] - np.mean(morlet_psd[channel]))
##########                            mask_buffer = (artifact_mask_buffer == 1)
##########                            morlet_psd[mask_buffer] = 1

                    # ========= PLOT PSD OLD =========
                    for channel in range(0, len(morlet_psd)):
                        self.plot_data_psd_eeg[channel] = morlet_psd[channel] - np.mean(morlet_psd[channel])
##                        self.plot_data_psd[channel] = np.sqrt(self.plot_data_psd[channel]) / 10
##                        self.plot_data_psd[channel] = self.plot_data_psd[channel] / 1000

##                    # === control for filter edge artifacts at begining of file
##                    if self.current_filt != 0:
##                        if self.total_samples_eeg < len(self.data_eeg):
##                            range_max_buffer_eeg = len(self.data_eeg) - self.total_samples_eeg + (self.bf_eeg_index_offset[self.current_filt] * 2)
##                            if range_max_buffer_eeg <= len(self.data_eeg):
##                                for channel in range(0, len(morlet_psd)):
##                                    self.plot_data_psd_eeg[channel, :, 0:int(range_max_buffer_eeg)] = np.mean(morlet_psd[channel])
##                            else:
##                                for channel in range(0, len(morlet_psd)):
##                                    self.plot_data_psd_eeg[channel, :, :] = np.mean(morlet_psd[channel])

                    # update line/image data
                    self.image_data_eeg = self._update_image_data_psd_eeg()
##                    self.image_data_accel = self._update_image_data_psd()
##                    print('local_clock/timestamp_eeg/diff: ' + str(local_clock()) + ', ' + str(timestamps_eeg[len(timestamps_eeg) - 1]) + ', ' + str(local_clock() - timestamps_eeg[len(timestamps_eeg) - 1]))

                # === EEG TF: SPECTROGRAM BANDPOWER WITH ARTIFACT REJECT
                self.last_time_test = local_clock()
                if self.toggle_device_view in ["EEG_TF_Band", "EEG_TF_BandChan"]:

##                    print('EEG TF: self.corner_widget_bool: ' + str(self.corner_widget_bool))
                    
                    # === RAW BANDPOWER: INDEX artifact time correction
                    total_points_buffer = int(len(timestamps_eeg) / 12)
##                    print('total_points_buffer: ' + str(total_points_buffer))
                    self.data_eeg_tf = np.roll(self.data_eeg_tf, -total_points_buffer, axis=1)
                    self.data_eeg_tf_filter = np.roll(self.data_eeg_tf_filter, -total_points_buffer, axis=2)
                    # === Bandpower Specs: window_eeg_tf: 640, window_eeg_tf_trim: 768, self.n_samples_tf_window = int(sampling_rate_eeg * 3) = 768
                    self.n_samples_tf_window = int((sampling_rate_eeg * 3) + np.max(self.bf_eeg_index_offset))
                    tf_artifact_data_eeg_range = [len(self.data_eeg) - self.n_samples_tf_window, len(self.data_eeg)]
                    tf_artifact_data_eeg = np.swapaxes(self.data_eeg[tf_artifact_data_eeg_range[0]:tf_artifact_data_eeg_range[1]], 0, 1)
                    tf_artifact_lsl_eeg = self.lsl_eeg[tf_artifact_data_eeg_range[0]:tf_artifact_data_eeg_range[1]]

                    # perform artifact rejection on self.n_samples_tf_window (e.g., 3 seconds) + np.max(bf)
                    tf_artifact_time_eeg, tf_artifact_mask_eeg = artifact_rejection_eeg(tf_artifact_data_eeg, sampling_rate_eeg)
                    # now trim long window to get raw eeg bandpower artifact_time to send into get_bandpower_scipy_spectrogram_2023
                    self.n_samples_tf_window_trim = int(sampling_rate_eeg * 3)
                    tf_artifact_data_eeg_range_trim = [len(self.data_eeg) - self.n_samples_tf_window_trim, len(self.data_eeg)]
                    tf_artifact_data_eeg_trim = np.swapaxes(self.data_eeg[tf_artifact_data_eeg_range_trim[0]:tf_artifact_data_eeg_range_trim[1]], 0, 1)
                    tf_artifact_lsl_eeg_trim = self.lsl_eeg[tf_artifact_data_eeg_range_trim[0]:tf_artifact_data_eeg_range_trim[1]]
                    tf_artifact_mask_eeg_trim = tf_artifact_mask_eeg[:, -self.n_samples_tf_window_trim:]
                    tf_artifact_time_eeg_trim = get_artifact_time_from_mask_2023(tf_artifact_mask_eeg_trim)



                    # === RAW NO FILT BANDPOWER Interpolation and No Interpolation, using get_bandpower_scipy_spectrogram_2023 to get final bandpower_artifact_mask as well
                    if self.corner_widget_bool == False:
                        tf_bandpower_overtime, tf_time_lsl, tf_time_index = get_bandpower_scipy_spectrogram_2023(
                            tf_artifact_data_eeg_trim, tf_artifact_lsl_eeg_trim, self.band_range_eeg, sampling_rate_eeg, nfft_eeg, 32)
                        tf_artifact_mask = np.zeros([n_chan_eeg, len(tf_time_index)])
##                    elif self.interp_toggle == "interp_off" and self.artifact_toggle != "artifact_off":
##                        tf_bandpower_overtime, tf_time_lsl, tf_time_index, tf_artifact_mask = get_bandpower_scipy_spectrogram_interp_2023_NEW(
##                            tf_artifact_data_eeg_trim, tf_artifact_lsl_eeg_trim, self.band_range_eeg, sampling_rate_eeg, nfft_eeg, 32, 'mask', tf_artifact_time_eeg_trim, "raw")
                    else:
                        tf_bandpower_overtime, tf_time_lsl, tf_time_index, tf_artifact_mask = get_bandpower_scipy_spectrogram_interp_2023_NEW(
                            tf_artifact_data_eeg_trim, tf_artifact_lsl_eeg_trim, self.band_range_eeg, sampling_rate_eeg, nfft_eeg, 32, 'mask', tf_artifact_time_eeg_trim, "interp")
##                            tf_artifact_data_eeg_trim, tf_artifact_lsl_eeg_trim, self.band_range_eeg, sampling_rate_eeg, nfft_eeg, 32, 'median', tf_artifact_time_eeg_trim, "interp")

                    # === EEG TF LAG ROLL WITH DATA ROLL
                    # === nfft_eeg, nperseg_eeg = 256, 64 defines bandpower_overtime time step of 58 samples (227 ms)
                    # === average bandpower over last 290 samples (58 * 5), e.g., 1136 ms (290 * 5) is calculated for each chan/band every 12 samples (lsl_chunk_length) using last self.n_samples_tf_window samples of data = 3 seconds
                    bandpower_timepoints = 5        # 290 samples (58 * 5), 1136 ms (290 * 5)
                    for value in range(0, total_points_buffer):
                        input_range_buffer = [int(len(tf_bandpower_overtime[0][0]) - bandpower_timepoints - value), int(len(tf_bandpower_overtime[0][0]) - value)]
                        average_bandpower_recent_buffer = tf_bandpower_overtime[:, :, input_range_buffer[0]:input_range_buffer[1]]
                        average_bandpower_recent = np.mean(average_bandpower_recent_buffer, axis = 2)
                        self.data_eeg_tf[:, (len(self.data_eeg_tf[0]) - value - 1), :] = average_bandpower_recent

                    # === EEG TF LAG ROLL WITH DATA ROLL BANDPOWER FILTER
                    for filt in range(0, len(self.eeg_filter_list_range)):
                        tf_artifact_data_eeg_range_filt = [len(self.data_eeg_filter_all[filt]) - self.n_samples_tf_window_trim, len(self.data_eeg_filter_all[filt])]
                        tf_artifact_data_eeg_filt = np.swapaxes(self.data_eeg_filter_all[filt][tf_artifact_data_eeg_range_filt[0]:tf_artifact_data_eeg_range_filt[1]], 0, 1)
                        tf_artifact_lsl_eeg_filt = self.lsl_eeg[tf_artifact_data_eeg_range_filt[0]:tf_artifact_data_eeg_range_filt[1]]
                        artifact_mask_range_buf = [int(len(tf_artifact_mask_eeg[0]) - self.n_samples_tf_window_trim - self.bf_eeg_index_offset[filt+1]), int(len(tf_artifact_mask_eeg[0]) - self.bf_eeg_index_offset[filt+1])]
                        tf_artifact_mask_eeg_filt = tf_artifact_mask_eeg[:, artifact_mask_range_buf[0]:artifact_mask_range_buf[1]]
                        tf_artifact_time_eeg_filt = get_artifact_time_from_mask_2023(tf_artifact_mask_eeg_filt)

                        # === FILTER BANDPOWER: Interpolation and No interpolation
                        if self.corner_widget_bool == False:
                            tf_bandpower_overtime_filt, tf_time_lsl_filt, tf_time_index_filt = get_bandpower_scipy_spectrogram_2023(
                                tf_artifact_data_eeg_filt, tf_artifact_lsl_eeg_filt, self.band_range_eeg, sampling_rate_eeg, nfft_eeg, 32)
##                                tf_artifact_data_eeg_filt, tf_artifact_lsl_eeg_filt, self.band_range_eeg, sampling_rate_eeg, nfft_eeg, nperseg_eeg)
                            tf_artifact_mask_filt = np.zeros([n_chan_eeg, len(tf_time_index_filt)])
##                        elif self.interp_toggle == "interp_off" and self.artifact_toggle != "artifact_off":
##                            tf_bandpower_overtime_filt, tf_time_lsl_filt, tf_time_index_filt, tf_artifact_mask_filt = get_bandpower_scipy_spectrogram_interp_2023_NEW(
##                                tf_artifact_data_eeg_filt, tf_artifact_lsl_eeg_filt, self.band_range_eeg, sampling_rate_eeg, nfft_eeg, 32, 'mask', tf_artifact_time_eeg_filt, "raw")
                        else:
                            tf_bandpower_overtime_filt, tf_time_lsl_filt, tf_time_index_filt, tf_artifact_mask_filt = get_bandpower_scipy_spectrogram_interp_2023_NEW(
                                tf_artifact_data_eeg_filt, tf_artifact_lsl_eeg_filt, self.band_range_eeg, sampling_rate_eeg, nfft_eeg, 32, 'mask', tf_artifact_time_eeg_filt, "interp")
##                                tf_artifact_data_eeg_filt, tf_artifact_lsl_eeg_filt, self.band_range_eeg, sampling_rate_eeg, nfft_eeg, 32, 'median', tf_artifact_time_eeg_filt, "interp")

                        # === EEG TF LAG ROLL WITH DATA ROLL
                        for value in range(0, total_points_buffer):
                            input_range_buffer_filt = [int(len(tf_bandpower_overtime_filt[0][0]) - bandpower_timepoints - value), int(len(tf_bandpower_overtime_filt[0][0]) - value)]
                            average_bandpower_recent_buffer_filt = tf_bandpower_overtime_filt[:, :, input_range_buffer_filt[0]:input_range_buffer_filt[1]]
                            average_bandpower_recent_filt = np.mean(average_bandpower_recent_buffer_filt, axis = 2)
                            self.data_eeg_tf_filter[filt, :, (len(self.data_eeg_tf_filter[filt][0]) - value - 1), :] = average_bandpower_recent_filt
                    # === BANDPOWER FILTER CHOICE
                    self.current_filt = self.filter_list_text_eeg.index(self.toggle_filt_eeg)
                    if self.current_filt == 0:
                        self.plot_data_eeg_tf_multiplot = self.data_eeg_tf
                    else:
                        self.plot_data_eeg_tf_multiplot = self.data_eeg_tf_filter[self.current_filt - 1]
                    # === TF Multi Plot: Separate Band, Separate Channel
                    if self.toggle_device_view == "EEG_TF_BandChan":
                        self._line_data_eeg_tf_multiplot = self._update_line_data_eeg_tf_multiplot()
                    # === TF Single Plot: Separate Band, Average Channel
                    if self.toggle_device_view == "EEG_TF_Band":
                        self.plot_data_eeg_tf_singleplot = np.swapaxes(self.plot_data_eeg_tf_multiplot.mean(axis = 2), 0, 1)
                        self._line_data_eeg_tf_singleplot = self._update_line_data_eeg_tf_singleplot()


####                # === EEG TF SIMPLIFIED USING getBandpower_2023
####                if self.toggle_device_view == 'EEG_TF_Band' or self.toggle_device_view == 'EEG_TF_BandChan':
####                    window_eeg_tf_trim = int(sampling_rate_eeg * 3)
####                    # === LAG ROLL WITH DATA ROLL
####                    eeg_tf_lag = int(len(timestamps_eeg) / 12)
####                    self.data_eeg_tf = np.roll(self.data_eeg_tf, -eeg_tf_lag, axis = 1)
####                    self.data_eeg_tf_filter = np.roll(self.data_eeg_tf_filter, -eeg_tf_lag, axis = 2)
####                    for value in range(0, eeg_tf_lag):
####                        input_range_buffer = [int(len(self.data_eeg) - window_eeg_tf_trim - (value * 12)), int(len(self.data_eeg) - (value * 12))]
####                        input_data_eeg_tf = np.swapaxes(self.data_eeg[input_range_buffer[0]:input_range_buffer[1], :], 0, 1)
####                        bandpower = np.swapaxes(getBandpower_2023(input_data_eeg_tf, nperseg_eeg, nfft_eeg, sampling_rate_eeg, self.band_range_eeg), 0, 1)
####                        self.data_eeg_tf[:, (len(self.data_eeg_tf[0]) - value - 1), :] = bandpower
####                        # === LAG ROLL WITH DATA ROLL FILTER
####                        for filt in range(0, len(self.eeg_filter_list_range)):
####                            input_data_eeg_tf_filt = np.swapaxes(self.data_eeg_filter_all[filt][-window_eeg_tf_trim:], 0, 1)
####                            bandpower_filt = np.swapaxes(getBandpower_2023(input_data_eeg_tf_filt, nperseg_eeg, nfft_eeg, sampling_rate_eeg, self.band_range_eeg), 0, 1)
####                            self.data_eeg_tf_filter[filt, :, (len(self.data_eeg_tf_filter[filt][0]) - 1), :] = bandpower_filt
####                            input_data_buffer_filt = self.data_eeg_filter_all[filt]
####                            input_range_buffer_filt = [int(len(input_data_buffer_filt) - window_eeg_tf_trim - (value * 12)), int(len(input_data_buffer_filt) - (value * 12))]
####                            input_data_eeg_tf_filt = np.swapaxes(input_data_buffer_filt[input_range_buffer[0]:input_range_buffer[1], :], 0, 1)
####                            bandpower_filt = np.swapaxes(getBandpower_2023(input_data_eeg_tf_filt, nperseg_eeg, nfft_eeg, sampling_rate_eeg, self.band_range_eeg), 0, 1)
####                            self.data_eeg_tf_filter[filt, :, (len(self.data_eeg_tf_filter[filt][0]) - value - 1), :] = bandpower_filt
####                    # update data
####                    self.current_filt = self.filter_list_text_eeg.index(self.toggle_filt_eeg)
####                    if self.current_filt == 0:
######                        self.plot_data_eeg_tf_multiplot = self.data_eeg_tf
####                        self.plot_data_eeg_tf_multiplot = self.data_eeg_tf
####                    else:
######                        self.plot_data_eeg_tf_multiplot = self.data_eeg_tf_filter[self.current_filt_eeg - 1]
####                        self.plot_data_eeg_tf_multiplot = self.data_eeg_tf_filter[self.current_filt - 1]
####                    # === TF Multi Plot: Separate Band, Separate Channel
####                    if self.toggle_device_view == "EEG_TF_BandChan":
####                        self._line_data_eeg_tf_multiplot = self._update_line_data_eeg_tf_multiplot()
####                    # === TF Single Plot: Separate Band, Average Channel
####                    if self.toggle_device_view == "EEG_TF_Band":
####                        self.plot_data_eeg_tf_singleplot = np.swapaxes(self.plot_data_eeg_tf_multiplot.mean(axis = 2), 0, 1)
####                        self._line_data_eeg_tf_singleplot = self._update_line_data_eeg_tf_singleplot()
######                # time testing
######                print('BANDPOWER END: Time Diff: ' + str(local_clock() - self.last_time_test) + ', total_samps: ' + str((1 / 256) / (local_clock() - self.last_time_test)))
######                self.last_time_test = local_clock()

                # === EEG ALL: EMIT DATA: All Device Line Data_Dict
                data_dict = {"toggle_view": self.toggle_device_view, "cam_toggle": self.cam_toggle, "line_toggles": self.line_bool_data_source, "play_bool": self.play_bool, "corner_widget_bool": self.corner_widget_bool,
                             "cam_toggle_count": self.cam_toggle_count,
##                             "artifact_pos_all_bandpower": self.artifact_pos_all_bandpower, "artifact_pos_all_bandpower_filt": self.artifact_pos_all_bandpower_filt,
                             "EEG_PSD": self.image_data_eeg, "ACCEL_PSD": self.image_data_accel, "GYRO_PSD": self.image_data_gyro, "PPG_PSD": self.image_data_ppg,
                             "current_artifact_filt_eeg": self.filter_list_text_eeg.index(self.toggle_filt_eeg), 
                             "timestamps_len": len(timestamps_eeg), "artifact_pos_all": self.artifact_pos_all,"bf_eeg_index_offset": self.bf_eeg_index_offset, 
                             "EEG_Time_Series": self._line_data_eeg, "ACCEL_Time_Series": self._line_data_accel, "GYRO_Time_Series": self._line_data_gyro, "PPG_Time_Series": self._line_data_ppg,
                             "EEG_FFT": self._line_data_eeg_fft, "ACCEL_FFT": self._line_data_accel_fft, "GYRO_FFT": self._line_data_gyro_fft, "PPG_FFT": self._line_data_ppg_fft,
                             "EEG_TF_Band": self._line_data_eeg_tf_singleplot, "ACCEL_TF_Band": self._line_data_accel_tf_singleplot, "GYRO_TF_Band": self._line_data_gyro_tf_singleplot, "PPG_TF_Band": self._line_data_ppg_tf_singleplot, 
                             "EEG_TF_BandChan": self._line_data_eeg_tf_multiplot, "ACCEL_TF_BandChan": self._line_data_accel_tf_multiplot, "GYRO_TF_BandChan": self._line_data_gyro_tf_multiplot, "PPG_TF_BandChan": self._line_data_ppg_tf_multiplot}
                self.new_data.emit(data_dict)
                self.artifact_pos_all = [[], [], [], []]

        # === REFACTOR ACCEL TREE
        if self.toggle_device_view in ['ACCEL_Time_Series', 'ACCEL_FFT', 'ACCEL_TF_Band', 'ACCEL_TF_BandChan', 'ACCEL_PSD']:
##            samples_accel, timestamps_accel = inlet_accel.pull_chunk(timeout=0.0, max_samples=24)
            samples_accel, timestamps_accel = inlet_accel.pull_chunk(timeout=0.0, max_samples=self.max_samples_accel)
            if timestamps_accel:
                self.lsl_accel = np.roll(self.lsl_accel, -len(timestamps_accel))
                self.lsl_accel[(len(self.lsl_accel) - len(timestamps_accel)):len(self.lsl_accel)] = timestamps_accel
                samples_accel = np.swapaxes(np.swapaxes(samples_accel, 0, 1)[:n_chan_accel], 0, 1)
                samples_accel = np.array(samples_accel)[:, ::-1]                # reverses channels
                # if this is first accel sample, replace all values with mean of first sample to scale camera window
                if np.sum(self.data_accel) == 0:
                    accel_mean_buffer = np.mean(np.array(samples_accel), axis=0)
                    for channel in range(0, n_chan_gyro):
                        self.data_accel[:, channel] = accel_mean_buffer[channel]
                # update self.data_accel with new samples_accel
                self.data_accel = np.vstack([self.data_accel, samples_accel])     # adds newest samples to end of len(self.data_accel)
                self.data_accel = self.data_accel[-window_accel:]              # removes oldest samples at start of self.data_accel0
                # === ARTIFACT update accel artifact_mask tracking
                self.full_artifact_mask_accel = np.roll(self.full_artifact_mask_accel, -len(timestamps_accel), axis=1)
                self.full_artifact_mask_accel[:, -len(timestamps_accel):] = 0
                # === 2023 ACCEL Filtering
                self.current_filt_accel = self.filter_list_text_accel.index(self.toggle_filt_accel)
                filt_samples_accel_buffer = np.zeros((len(self.accel_filter_list_range), len(samples_accel), n_chan_accel))
                for filt in range(0, len(self.accel_filter_list_range)):
                    filt_samples_accel_buffer[filt], self.filt_state_accel_all[filt] = lfilter(self.bf_accel_all[filt], self.af, samples_accel, axis=0, zi=self.filt_state_accel_all[filt])
                    stack_data_buffer = np.vstack([self.data_accel_filter_all[filt], filt_samples_accel_buffer[filt]])
                    self.data_accel_filter_all[filt] = stack_data_buffer[-window_accel:]
                # update data
                if self.current_filt_accel == 0:
##                    if np.sum(self.data_accel[0:(len(self.data_accel) - len(timestamps_accel))]) == 0:
##                        self.data_accel[0:(len(self.data_accel) - len(timestamps_accel))] = self.data_accel[len(self.data_accel) - 1]
####                    self.plot_data_accel = ((self.data_accel - self.data_accel.mean(axis=0)))
                    self.plot_data_accel = ((self.data_accel - self.data_accel.mean(axis=0)) * 10)                  # scale accel for plotting
                else:
##                    self.plot_data_accel = (self.data_accel_filter_all[self.current_filt_accel - 1])
                    self.plot_data_accel = (self.data_accel_filter_all[self.current_filt_accel - 1] * 10)           # scale accel for plotting

##                # === control for filter edge artifacts at begining of file
##                self.total_samples_accel += len(timestamps_accel)
####                range_max_buffer_accel = len(self.data_accel) - self.total_samples_accel + (np.max(self.bf_accel_index_offset) * 2)
##                if self.current_filt_accel != 0:
##                    if self.total_samples_accel < len(self.data_accel):
##                        range_max_buffer_accel = len(self.data_accel) - self.total_samples_accel + (self.bf_accel_index_offset[self.current_filt_accel] * 2)
##                        if range_max_buffer_accel <= len(self.data_accel):
##                            self.plot_data_accel[0:int(range_max_buffer_accel), :] = 0
##                        else:
##                            self.plot_data_accel[:, :] = 0

                # === ARTIFACT REJECTION from PetalDisplay_v9.py
                if self.corner_widget_bool == True:
                    # window = 1024 = 4 seconds: since max(len(self.bf_eeg_all)) = 422 and 512 window doesn't miss samples, artifact detect in last 512 + 512 = 4 seconds
                    # interval = 256 = 1 second: perform artifact detection every second for window = last 4 seconds of data
                    artifact_interval = 1                               # perform artifact rejection every artifact_interval seconds
                    artifact_window_accel = int(sampling_rate_accel * 4)        # artifact window = 1024 = 4 seconds
                    if (local_clock() - self.artifact_detect_lsl_accel) > artifact_interval:
                        # define start/end sample index for artifact_data_eeg with last artifact_window samples of data
                        artifact_data_accel_range = [len(self.data_accel) - artifact_window_accel, len(self.data_accel)]
                        artifact_data_accel = np.swapaxes(self.data_accel[artifact_data_accel_range[0]:artifact_data_accel_range[1]], 0, 1)
                        artifact_time_accel, artifact_mask_accel = artifact_rejection_accel(artifact_data_accel, sampling_rate_accel)
                        # get artifact_time_eeg_index for raw and artifact_time_eeg_index_corrected for filter time/index correction
                        artifact_time_accel_index = get_artifact_time_index_raw(artifact_time_accel, artifact_data_accel_range)
                        artifact_time_accel_index_corrected = get_artifact_time_index_corrected(artifact_time_accel, artifact_data_accel_range, self.bf_accel_index_offset)
                        self.full_artifact_mask_accel = update_full_artifact_mask(self.full_artifact_mask_accel, artifact_time_accel_index, self.bf_accel_index_offset)
                        # === ARTIFACT LINES NEW
                        if len(artifact_time_accel_index) > 0:
                            self.artifact_pos_all_accel = artifact_time_accel_index
                        else:
                            self.artifact_pos_all_accel = [[], [], []]
                        # === reset artifact detection lsl clock
                        self.artifact_detect_lsl_accel = local_clock()

                        # UPDATE CornerWidget
                        accel_artifact_test = ""
                        for channel in range(0, len(self.artifact_pos_all_accel)):
                            if len(self.artifact_pos_all_accel[channel]) > 0:
                                if accel_artifact_test == "":
                                    accel_artifact_test = str(ch_names_accel[channel])
                                else:
                                    accel_artifact_test = accel_artifact_test + ", " + str(ch_names_accel[channel])
                        if accel_artifact_test == "":
                            accel_artifact_test = "None"
                        data_dict_corner_widget = {"heart_bpm": "None", "eeg_artifact": "None", "accel_movement": str(accel_artifact_test), "gyro_movement": "None"}
                        self.new_data_corner_widget.emit(data_dict_corner_widget)

                # === UPDATE LINE ACCEL data
                self._line_data_accel = self._update_line_data_accel()

                # === ACCEL PSD
                if self.toggle_device_view == "ACCEL_PSD":
                    if self.current_filt_accel == 0:
                        tf_data_buffer_accel = (self.data_accel - self.data_accel.mean(axis=0))
                        tf_data_buffer_accel = np.array([np.swapaxes(tf_data_buffer_accel, 0, 1)])
##                        test_data = (self.data_accel - self.data_accel.mean(axis=0))
##                        test_data = np.array([np.swapaxes(test_data, 0, 1)])
                    else:
                        tf_data_buffer_accel = np.array([np.swapaxes(self.data_accel_filter_all[self.current_filt_accel - 1], 0, 1)])
##                        test_data = np.array([np.swapaxes(self.data_accel_filter_all[self.current_filt_accel - 1], 0, 1)])
                    morlet_psd_accel = tfr_array_morlet(tf_data_buffer_accel, sfreq=sampling_rate_accel, freqs=freqs_psd_accel, output='power')[0]
##                    morlet_test = tfr_array_morlet(test_data, sfreq=sampling_rate_accel, freqs=freqs_psd_accel, output='power')[0]
##                    for channel in range(0, len(morlet_test)):
##                        self.plot_data_psd_accel[channel] = morlet_test[channel] - np.mean(morlet_test[channel])
                    for channel in range(0, len(morlet_psd_accel)):
                        self.plot_data_psd_accel[channel] = morlet_psd_accel[channel] - np.mean(morlet_psd_accel[channel])

##                    # === control for filter edge artifacts at begining of file
##                    if self.current_filt_accel != 0:
##                        if self.total_samples_accel < len(self.data_accel):
##                            range_max_buffer_accel = len(self.data_accel) - self.total_samples_accel + (self.bf_accel_index_offset[self.current_filt_accel] * 2)
##                            if range_max_buffer_accel <= len(self.data_accel):
##                                for channel in range(0, len(morlet_psd_accel)):
##                                    self.plot_data_psd_accel[channel, :, 0:int(range_max_buffer_accel)] = np.mean(morlet_psd_accel[channel])
##                            else:
##                                for channel in range(0, len(morlet_psd_accel)):
##                                    self.plot_data_psd_accel[channel, :, :] = np.mean(morlet_psd_accel[channel])

                    # update line/image data
                    self.image_data_accel = self._update_image_data_psd_accel()
##                    print('local_clock/timestamp_eeg/diff: ' + str(local_clock()) + ', ' + str(timestamps_eeg[len(timestamps_eeg) - 1]) + ', ' + str(local_clock() - timestamps_eeg[len(timestamps_eeg) - 1]))

                # === ACCEL FFT
                if self.toggle_device_view == "ACCEL_FFT":
                    data_accel_fft = []
                    total_samples_accel_fft = 52
                    input_data_accel_fft = np.swapaxes(self.plot_data_accel, 0, 1)
                    for channel in range(0, n_chan_accel_fft):
                        fft_data_buffer, fft_freq_buffer = getFFT(input_data_accel_fft[channel][-window_accel_fft_trim:], sampling_rate_accel)
                        data_accel_fft.append(fft_data_buffer)
                    data_accel_fft, freq_accel_fft = np.asarray(data_accel_fft), fft_freq_buffer
                    self.data_accel_fft = data_accel_fft
                    self.plot_data_accel_fft = (np.swapaxes(self.data_accel_fft, 0, 1)) * 100
##                    self.plot_data_accel_fft = (np.swapaxes(self.data_accel_fft, 0, 1))
                    self._line_data_accel_fft = self._update_line_data_accel_fft()

                # === ACCEL TF
                if self.toggle_device_view in ['ACCEL_TF_Band', 'ACCEL_TF_BandChan']:
                    # === ACCEL TF LAG ROLL WITH DATA ROLL
                    accel_tf_lag = int(len(timestamps_accel) / 4)
                    self.data_accel_tf = np.roll(self.data_accel_tf, -accel_tf_lag, axis = 1)
                    self.data_accel_tf_filter = np.roll(self.data_accel_tf_filter, -accel_tf_lag, axis = 2)
                    for value in range(0, accel_tf_lag):
                        input_range_buffer = [int(len(self.data_accel) - window_accel_tf_trim - (value * 4)), int(len(self.data_accel) - (value * 4))]
                        input_data_accel_tf = np.swapaxes(self.data_accel[input_range_buffer[0]:input_range_buffer[1], :], 0, 1)
                        bandpower = np.swapaxes(getBandpower_2023(input_data_accel_tf, nperseg_accel, nfft_accel, sampling_rate_accel, self.band_range_accel), 0, 1)
                        self.data_accel_tf[:, (len(self.data_accel_tf[0]) - value - 1), :] = bandpower
                        # === LAG ROLL WITH DATA ROLL FILTER
                        for filt in range(0, len(self.accel_filter_list_range)):
                            input_data_accel_tf_filt = np.swapaxes(self.data_accel_filter_all[filt][-window_accel_tf_trim:], 0, 1)
                            bandpower_filt = np.swapaxes(getBandpower_2023(input_data_accel_tf_filt, nperseg_accel, nfft_accel, sampling_rate_accel, self.band_range_accel), 0, 1)
                            self.data_accel_tf_filter[filt, :, (len(self.data_accel_tf_filter[filt][0]) - 1), :] = bandpower_filt
                            input_data_buffer_filt = self.data_accel_filter_all[filt]
                            input_range_buffer_filt = [int(len(input_data_buffer_filt) - window_accel_tf_trim - (value * 4)), int(len(input_data_buffer_filt) - (value * 4))]
                            input_data_accel_tf_filt = np.swapaxes(input_data_buffer_filt[input_range_buffer[0]:input_range_buffer[1], :], 0, 1)
                            bandpower_filt = np.swapaxes(getBandpower_2023(input_data_accel_tf_filt, nperseg_accel, nfft_accel, sampling_rate_accel, self.band_range_accel), 0, 1)
                            self.data_accel_tf_filter[filt, :, (len(self.data_accel_tf_filter[filt][0]) - value - 1), :] = bandpower_filt

                    # update data
##                    self.current_filt = self.filter_list_text_accel.index(self.toggle_filt_accel)
                    self.current_filt_accel_buffer = self.filter_list_text_accel.index(self.toggle_filt_accel)
                    if self.current_filt_accel == 0:
##                        self.plot_data_accel_tf_multiplot = self.data_accel_tf
                        self.plot_data_accel_tf_multiplot = self.data_accel_tf
                    else:
##                        self.plot_data_accel_tf_multiplot = self.data_accel_tf_filter[self.current_filt_accel - 1]
                        self.plot_data_accel_tf_multiplot = self.data_accel_tf_filter[self.current_filt_accel_buffer - 1]
                    # === TF Multi Plot: Separate Band, Separate Channel
                    if self.toggle_device_view == "ACCEL_TF_BandChan":
                        self._line_data_accel_tf_multiplot = self._update_line_data_accel_tf_multiplot()
                    # === TF Single Plot: Separate Band, Average Channel
                    if self.toggle_device_view == "ACCEL_TF_Band":
                        self.plot_data_accel_tf_singleplot = np.swapaxes(self.plot_data_accel_tf_multiplot.mean(axis = 2), 0, 1)
                        self._line_data_accel_tf_singleplot = self._update_line_data_accel_tf_singleplot()

                # === ACCEL ALL: EMIT DATA: All Device Line Data_Dict
                data_dict = {"toggle_view": self.toggle_device_view, "cam_toggle": self.cam_toggle, "line_toggles": self.line_bool_data_source,
                             "cam_toggle_count": self.cam_toggle_count,
                             "timestamps_len": len(timestamps_accel), "artifact_pos_all": self.artifact_pos_all,"bf_eeg_index_offset": self.bf_eeg_index_offset,
                             "artifact_pos_all_accel": self.artifact_pos_all_accel,"bf_accel_index_offset": self.bf_accel_index_offset,
                             "current_artifact_filt_accel": self.filter_list_text_accel.index(self.toggle_filt_accel),
                             "current_artifact_filt_eeg": self.filter_list_text_eeg.index(self.toggle_filt_eeg), "play_bool": self.play_bool, "corner_widget_bool": self.corner_widget_bool,  
                             "EEG_PSD": self.image_data_eeg, "ACCEL_PSD": self.image_data_accel, "GYRO_PSD": self.image_data_gyro, "PPG_PSD": self.image_data_ppg,
                             "EEG_Time_Series": self._line_data_eeg, "ACCEL_Time_Series": self._line_data_accel, "GYRO_Time_Series": self._line_data_gyro, "PPG_Time_Series": self._line_data_ppg,
                             "EEG_FFT": self._line_data_eeg_fft, "ACCEL_FFT": self._line_data_accel_fft, "GYRO_FFT": self._line_data_gyro_fft, "PPG_FFT": self._line_data_ppg_fft,
                             "EEG_TF_Band": self._line_data_eeg_tf_singleplot, "ACCEL_TF_Band": self._line_data_accel_tf_singleplot, "GYRO_TF_Band": self._line_data_gyro_tf_singleplot, "PPG_TF_Band": self._line_data_ppg_tf_singleplot, 
                             "EEG_TF_BandChan": self._line_data_eeg_tf_multiplot, "ACCEL_TF_BandChan": self._line_data_accel_tf_multiplot, "GYRO_TF_BandChan": self._line_data_gyro_tf_multiplot, "PPG_TF_BandChan": self._line_data_ppg_tf_multiplot}
                self.new_data.emit(data_dict)
                self.artifact_pos_all_accel = [[], [], []]

        # === REFACTOR GYRO TREE
        if self.toggle_device_view in ['GYRO_Time_Series', 'GYRO_FFT', 'GYRO_TF_Band', 'GYRO_TF_BandChan', 'GYRO_PSD']:
            samples_gyro, timestamps_gyro = inlet_gyro.pull_chunk(timeout=0.0, max_samples=self.max_samples_gyro)
##            samples_gyro, timestamps_gyro = inlet_gyro.pull_chunk(timeout=0.0, max_samples=24)
            if timestamps_gyro:
                self.lsl_gyro = np.roll(self.lsl_gyro, -len(timestamps_gyro))
                self.lsl_gyro[(len(self.lsl_gyro) - len(timestamps_gyro)):len(self.lsl_gyro)] = timestamps_gyro
                samples_gyro = np.swapaxes(np.swapaxes(samples_gyro, 0, 1)[:n_chan_gyro], 0, 1)
                samples_gyro = np.array(samples_gyro)[:, ::-1]                # reverses channels
                # if this is first gyro sample, replace all values with mean of first sample to scale camera window
                if np.sum(self.data_gyro) == 0:
                    gyro_mean_buffer = np.mean(np.array(samples_gyro), axis=0)
                    for channel in range(0, n_chan_gyro):
                        self.data_gyro[:, channel] = gyro_mean_buffer[channel]
                # update self.data_gyro with new samples_gyro
                self.data_gyro = np.vstack([self.data_gyro, samples_gyro])     # adds newest samples to end of len(self.data_gyro)
                self.data_gyro = self.data_gyro[-window_gyro:]              # removes oldest samples at start of self.data_gyro
                # === ARTIFACT update GYRO artifact_mask tracking
                self.full_artifact_mask_gyro = np.roll(self.full_artifact_mask_gyro, -len(timestamps_gyro), axis=1)
                self.full_artifact_mask_gyro[:, -len(timestamps_gyro):] = 0
                # === 2023 GYRO Filtering
                self.current_filt_gyro = self.filter_list_text_gyro.index(self.toggle_filt_gyro)
                filt_samples_gyro_buffer = np.zeros((len(self.gyro_filter_list_range), len(samples_gyro), n_chan_gyro))
                for filt in range(0, len(self.gyro_filter_list_range)):
                    filt_samples_gyro_buffer[filt], self.filt_state_gyro_all[filt] = lfilter(self.bf_gyro_all[filt], self.af, samples_gyro, axis=0, zi=self.filt_state_gyro_all[filt])
                    stack_data_buffer = np.vstack([self.data_gyro_filter_all[filt], filt_samples_gyro_buffer[filt]])
                    self.data_gyro_filter_all[filt] = stack_data_buffer[-window_gyro:]
                # set plot data eeg, swap data for FFT
                if self.current_filt_gyro == 0:
####                    if np.sum(self.data_gyro[0:(len(self.data_gyro) - len(timestamps_gyro))]) == 0:
####                        self.data_gyro[0:(len(self.data_gyro) - len(timestamps_gyro))] = self.data_gyro[len(self.data_gyro) - 1]
##                    self.plot_data_gyro = ((self.data_gyro - self.data_gyro.mean(axis=0)))
                    self.plot_data_gyro = ((self.data_gyro - self.data_gyro.mean(axis=0)) / 10)                 # scale gyro for plotting
                else:
##                    self.plot_data_gyro = (self.data_gyro_filter_all[self.current_filt_gyro - 1])
                    self.plot_data_gyro = (self.data_gyro_filter_all[self.current_filt_gyro - 1] / 10)          # scale gyro for plotting

##                # === control for filter edge artifacts at begining of file
##                self.total_samples_gyro += len(timestamps_gyro)
##                if self.current_filt_gyro != 0:
##                    if self.total_samples_gyro < len(self.data_gyro):
##                        range_max_buffer_gyro = len(self.data_gyro) - self.total_samples_gyro + (self.bf_gyro_index_offset[self.current_filt_gyro] * 2)
##                        if range_max_buffer_gyro <= len(self.data_gyro):
##                            self.plot_data_gyro[0:int(range_max_buffer_gyro), :] = self.plot_data_gyro[0:int(range_max_buffer_gyro), :] / 1000
##                        else:
##                            self.plot_data_gyro[:, :] = self.plot_data_gyro[:, :] / 1000

                # === ARTIFACT REJECTION from PetalDisplay_v9.py
                if self.corner_widget_bool == True:
                    # window = 1024 = 4 seconds: since max(len(self.bf_eeg_all)) = 422 and 512 window doesn't miss samples, artifact detect in last 512 + 512 = 4 seconds
                    # interval = 256 = 1 second: perform artifact detection every second for window = last 4 seconds of data
                    artifact_interval = 1                               # perform artifact rejection every artifact_interval seconds
                    artifact_window_gyro = int(sampling_rate_gyro * 4)        # artifact window = 1024 = 4 seconds
                    if (local_clock() - self.artifact_detect_lsl_gyro) > artifact_interval:
                        # define start/end sample index for artifact_data_eeg with last artifact_window samples of data
                        artifact_data_gyro_range = [len(self.data_gyro) - artifact_window_gyro, len(self.data_gyro)]
                        artifact_data_gyro = np.swapaxes(self.data_gyro[artifact_data_gyro_range[0]:artifact_data_gyro_range[1]], 0, 1)
                        artifact_time_gyro, artifact_mask_gyro = artifact_rejection_gyro(artifact_data_gyro, sampling_rate_gyro)
                        # get artifact_time_eeg_index for raw and artifact_time_eeg_index_corrected for filter time/index correction
                        artifact_time_gyro_index = get_artifact_time_index_raw(artifact_time_gyro, artifact_data_gyro_range)
                        artifact_time_gyro_index_corrected = get_artifact_time_index_corrected(artifact_time_gyro, artifact_data_gyro_range, self.bf_gyro_index_offset)
                        self.full_artifact_mask_gyro = update_full_artifact_mask(self.full_artifact_mask_gyro, artifact_time_gyro_index, self.bf_gyro_index_offset)
                        # === ARTIFACT LINES NEW
                        if len(artifact_time_gyro_index) > 0:
                            self.artifact_pos_all_gyro = artifact_time_gyro_index
                        else:
                            self.artifact_pos_all_gyro = [[], [], []]
                        # === reset artifact detection lsl clock
                        self.artifact_detect_lsl_gyro = local_clock()

                        # UPDATE CornerWidget
                        gyro_artifact_test = ""
                        for channel in range(0, len(self.artifact_pos_all_gyro)):
                            if len(self.artifact_pos_all_gyro[channel]) > 0:
                                if gyro_artifact_test == "":
                                    gyro_artifact_test = str(ch_names_gyro[channel])
                                else:
                                    gyro_artifact_test = gyro_artifact_test + ", " + str(ch_names_gyro[channel])
                        if gyro_artifact_test == "":
                            gyro_artifact_test = "None"
                        data_dict_corner_widget = {"heart_bpm": "None", "eeg_artifact": "None", "accel_movement": "None", "gyro_movement": str(gyro_artifact_test)}
                        self.new_data_corner_widget.emit(data_dict_corner_widget)

                # === UPDATE LINE GYRO data
                self._line_data_gyro = self._update_line_data_gyro()

                # === GYRO PSD
                if self.toggle_device_view == "GYRO_PSD":
                    if self.current_filt_gyro == 0:
                        tf_data_buffer_gyro = (self.data_gyro - self.data_gyro.mean(axis=0))
                        tf_data_buffer_gyro = np.array([np.swapaxes(tf_data_buffer_gyro, 0, 1)])
                    else:
                        tf_data_buffer_gyro = np.array([np.swapaxes(self.data_gyro_filter_all[self.current_filt_gyro - 1], 0, 1)])
                    morlet_psd_gyro = tfr_array_morlet(tf_data_buffer_gyro, sfreq=sampling_rate_gyro, freqs=freqs_psd_gyro, output='power')[0]
                    for channel in range(0, len(morlet_psd_gyro)):
                        self.plot_data_psd_gyro[channel] = morlet_psd_gyro[channel] - np.mean(morlet_psd_gyro[channel])

##                    # === control for filter edge artifacts at begining of file
##                    if self.current_filt_gyro != 0:
##                        if self.total_samples_gyro < len(self.data_gyro):
##                            range_max_buffer_gyro = len(self.data_gyro) - self.total_samples_gyro + (self.bf_gyro_index_offset[self.current_filt_gyro] * 2)
##                            if range_max_buffer_gyro <= len(self.data_gyro):
##                                for channel in range(0, len(morlet_psd_gyro)):
##                                    self.plot_data_psd_gyro[channel, :, 0:int(range_max_buffer_gyro)] = np.mean(morlet_psd_gyro[channel])
##                            else:
##                                for channel in range(0, len(morlet_psd_gyro)):
##                                    self.plot_data_psd_gyro[channel, :, :] = np.mean(morlet_psd_gyro[channel])

                    # update line/image data
                    self.image_data_gyro = self._update_image_data_psd_gyro()

                # === GYRO FFT
                if self.toggle_device_view == "GYRO_FFT":
                    data_gyro_fft = []
                    total_samples_gyro_fft = 52
                    input_data_gyro_fft = np.swapaxes(self.plot_data_gyro, 0, 1)
                    for channel in range(0, n_chan_gyro_fft):
                        fft_data_buffer, fft_freq_buffer = getFFT(input_data_gyro_fft[channel][-window_gyro_fft_trim:], sampling_rate_gyro)
                        data_gyro_fft.append(fft_data_buffer)
                    data_gyro_fft, freq_gyro_fft = np.asarray(data_gyro_fft), fft_freq_buffer
                    self.data_gyro_fft = data_gyro_fft
                    self.plot_data_gyro_fft = (np.swapaxes(self.data_gyro_fft, 0, 1)) * 100
##                    self.plot_data_gyro_fft = (np.swapaxes(self.data_gyro_fft, 0, 1))
                    self._line_data_gyro_fft = self._update_line_data_gyro_fft()

                # === GYRO TF
                if self.toggle_device_view in ['GYRO_TF_Band', 'GYRO_TF_BandChan']:
                    # === GYRO TF LAG ROLL WITH DATA ROLL
                    gyro_tf_lag = int(len(timestamps_gyro) / 4)
                    self.data_gyro_tf = np.roll(self.data_gyro_tf, -gyro_tf_lag, axis = 1)
                    self.data_gyro_tf_filter = np.roll(self.data_gyro_tf_filter, -gyro_tf_lag, axis = 2)
                    for value in range(0, gyro_tf_lag):
                        input_range_buffer = [int(len(self.data_gyro) - window_gyro_tf_trim - (value * 4)), int(len(self.data_gyro) - (value * 4))]
                        input_data_gyro_tf = np.swapaxes(self.data_gyro[input_range_buffer[0]:input_range_buffer[1], :], 0, 1)
                        bandpower = np.swapaxes(getBandpower_2023(input_data_gyro_tf, nperseg_gyro, nfft_gyro, sampling_rate_gyro, self.band_range_gyro), 0, 1)
                        self.data_gyro_tf[:, (len(self.data_gyro_tf[0]) - value - 1), :] = bandpower
                        # === LAG ROLL WITH DATA ROLL FILTER
                        for filt in range(0, len(self.gyro_filter_list_range)):
                            input_data_gyro_tf_filt = np.swapaxes(self.data_gyro_filter_all[filt][-window_gyro_tf_trim:], 0, 1)
                            bandpower_filt = np.swapaxes(getBandpower_2023(input_data_gyro_tf_filt, nperseg_gyro, nfft_gyro, sampling_rate_gyro, self.band_range_gyro), 0, 1)
                            self.data_gyro_tf_filter[filt, :, (len(self.data_gyro_tf_filter[filt][0]) - 1), :] = bandpower_filt
                            input_data_buffer_filt = self.data_gyro_filter_all[filt]
                            input_range_buffer_filt = [int(len(input_data_buffer_filt) - window_gyro_tf_trim - (value * 4)), int(len(input_data_buffer_filt) - (value * 4))]
                            input_data_gyro_tf_filt = np.swapaxes(input_data_buffer_filt[input_range_buffer[0]:input_range_buffer[1], :], 0, 1)
                            bandpower_filt = np.swapaxes(getBandpower_2023(input_data_gyro_tf_filt, nperseg_gyro, nfft_gyro, sampling_rate_gyro, self.band_range_gyro), 0, 1)
                            self.data_gyro_tf_filter[filt, :, (len(self.data_gyro_tf_filter[filt][0]) - value - 1), :] = bandpower_filt
                    # update data
##                    self.current_filt = self.filter_list_text_gyro.index(self.toggle_filt_gyro)
                    self.current_filt_gyro_buffer = self.filter_list_text_gyro.index(self.toggle_filt_gyro)
                    if self.current_filt_gyro == 0:
                        self.plot_data_gyro_tf_multiplot = self.data_gyro_tf
                    else:
##                        self.plot_data_gyro_tf_multiplot = self.data_gyro_tf_filter[self.current_filt_gyro - 1]
                        self.plot_data_gyro_tf_multiplot = self.data_gyro_tf_filter[self.current_filt_gyro_buffer - 1]
                    # === TF Multi Plot: Separate Band, Separate Channel
                    if self.toggle_device_view == "GYRO_TF_BandChan":
                        self._line_data_gyro_tf_multiplot = self._update_line_data_gyro_tf_multiplot()
                    # === TF Single Plot: Separate Band, Average Channel
                    if self.toggle_device_view == "GYRO_TF_Band":
                        self.plot_data_gyro_tf_singleplot = np.swapaxes(self.plot_data_gyro_tf_multiplot.mean(axis = 2), 0, 1)
                        self._line_data_gyro_tf_singleplot = self._update_line_data_gyro_tf_singleplot()

                # === GYRO ALL: EMIT DATA: All Device Line Data_Dict
                data_dict = {"toggle_view": self.toggle_device_view, "cam_toggle": self.cam_toggle, "line_toggles": self.line_bool_data_source,
                             "cam_toggle_count": self.cam_toggle_count,
                             "timestamps_len": len(timestamps_gyro), "artifact_pos_all": self.artifact_pos_all,"bf_eeg_index_offset": self.bf_eeg_index_offset,
                             "artifact_pos_all_gyro": self.artifact_pos_all_gyro, "bf_gyro_index_offset": self.bf_gyro_index_offset,
                             "current_artifact_filt_gyro": self.filter_list_text_gyro.index(self.toggle_filt_gyro),
                             "current_artifact_filt_eeg": self.filter_list_text_eeg.index(self.toggle_filt_eeg), "play_bool": self.play_bool, "corner_widget_bool": self.corner_widget_bool,  
                             "EEG_PSD": self.image_data_eeg, "ACCEL_PSD": self.image_data_accel, "GYRO_PSD": self.image_data_gyro, "PPG_PSD": self.image_data_ppg,
                             "EEG_Time_Series": self._line_data_eeg, "ACCEL_Time_Series": self._line_data_accel, "GYRO_Time_Series": self._line_data_gyro, "PPG_Time_Series": self._line_data_ppg,
                             "EEG_FFT": self._line_data_eeg_fft, "ACCEL_FFT": self._line_data_accel_fft, "GYRO_FFT": self._line_data_gyro_fft, "PPG_FFT": self._line_data_ppg_fft,
                             "EEG_TF_Band": self._line_data_eeg_tf_singleplot, "ACCEL_TF_Band": self._line_data_accel_tf_singleplot, "GYRO_TF_Band": self._line_data_gyro_tf_singleplot, "PPG_TF_Band": self._line_data_ppg_tf_singleplot, 
                             "EEG_TF_BandChan": self._line_data_eeg_tf_multiplot, "ACCEL_TF_BandChan": self._line_data_accel_tf_multiplot, "GYRO_TF_BandChan": self._line_data_gyro_tf_multiplot, "PPG_TF_BandChan": self._line_data_ppg_tf_multiplot}
                self.new_data.emit(data_dict)
                self.artifact_pos_all_gyro = [[], [], []]

        # === PPG TREE: CH1/CH2 TOGETHER, CH1 ALONE = ambient
        if self.toggle_device_view in ['PPG_Time_Series', 'PPG_FFT', 'PPG_TF_Band', 'PPG_TF_BandChan', 'PPG_PSD']:
            samples_ppg, timestamps_ppg = inlet_ppg.pull_chunk(timeout=0.0, max_samples=self.max_samples_ppg)
##            samples_ppg, timestamps_ppg = inlet_ppg.pull_chunk(timeout=0.0, max_samples=24)
            if timestamps_ppg:               
                self.lsl_ppg = np.roll(self.lsl_ppg, -len(timestamps_ppg))
                self.lsl_ppg[(len(self.lsl_ppg) - len(timestamps_ppg)):len(self.lsl_ppg)] = timestamps_ppg
                samples_ppg = np.swapaxes(np.swapaxes(samples_ppg, 0, 1)[:n_chan_ppg], 0, 1)
                samples_ppg = np.array(samples_ppg)[:, ::-1]                # reverses channels
                # if this is first ppg sample, replace all values with mean of first sample to scale camera window
                if np.sum(self.data_ppg) == 0:
                    ppg_mean_buffer = np.mean(np.array(samples_ppg), axis=0)
                    for channel in range(0, n_chan_ppg):
                        self.data_ppg[:, channel] = ppg_mean_buffer[channel]
                # update self.data_ppg with new samples_ppg
                self.data_ppg = np.vstack([self.data_ppg, samples_ppg])     # adds newest samples to end of len(self.data_ppg)
                self.data_ppg = self.data_ppg[-window_ppg:]              # removes oldest samples at start of self.data_ppg
                # === 2023 PPG Filtering
                self.current_filt_ppg = self.filter_list_text_ppg.index(self.toggle_filt_ppg)
                filt_samples_ppg_buffer = np.zeros((len(self.ppg_filter_list_range), len(samples_ppg), n_chan_ppg))
                for filt in range(0, len(self.ppg_filter_list_range)):
                    filt_samples_ppg_buffer[filt], self.filt_state_ppg_all[filt] = lfilter(self.bf_ppg_all[filt], self.af, samples_ppg, axis=0, zi=self.filt_state_ppg_all[filt])
                    stack_data_buffer = np.vstack([self.data_ppg_filter_all[filt], filt_samples_ppg_buffer[filt]])
                    self.data_ppg_filter_all[filt] = stack_data_buffer[-window_ppg:]
                # set plot data eeg, swap data for FFT
                if self.current_filt_ppg == 0:
####                    if np.sum(self.data_ppg[0:(len(self.data_ppg) - len(timestamps_ppg))]) == 0:
####                        self.data_ppg[0:(len(self.data_ppg) - len(timestamps_ppg))] = self.data_ppg[len(self.data_ppg) - 1]      
##                    self.plot_data_ppg = (self.data_ppg - self.data_ppg.mean(axis=0))
                    self.plot_data_ppg = (self.data_ppg - self.data_ppg.mean(axis=0)) / 1000                # scale ppg for plotting
                else:
##                    self.plot_data_ppg = (self.data_ppg_filter_all[self.current_filt_ppg - 1])
                    self.plot_data_ppg = (self.data_ppg_filter_all[self.current_filt_ppg - 1]) / 1000       # scale ppg for plotting

                # scale channel 1 separately
                self.plot_data_ppg[:, 0] = self.plot_data_ppg[:, 0] * 10

##                # === control for filter edge artifacts at begining of file
##                self.total_samples_ppg += len(timestamps_ppg)
####                print('self.bf_ppg_index_offset: ' + str(self.bf_ppg_index_offset))
##                if self.current_filt_ppg != 0:
##                    if self.total_samples_ppg < len(self.data_ppg):
##                        range_max_buffer_ppg = len(self.data_ppg) - self.total_samples_ppg + (self.bf_ppg_index_offset[self.current_filt_ppg] * 2)
##                        if range_max_buffer_ppg <= len(self.data_ppg):
##                            self.plot_data_ppg[0:int(range_max_buffer_ppg), :] = 0
##                        else:
##                            self.plot_data_ppg[:, :] = 0

                # update line data
                self._line_data_ppg = self._update_line_data_ppg()

                # === PPG BPM MIRROR OLD
                # get raw PPG input data for fft bpm calculation
##                fft_ppg_input_data = np.swapaxes(self.data_ppg, 0, 1)[1]
####                fft_ppg_input_data = np.swapaxes(self.plot_data_ppg, 0, 1)[1]
##                # mirror data +/- x5
##                fft_ppg_input_data_mirror = np.tile(np.append(fft_ppg_input_data, np.flip(fft_ppg_input_data, -1)), 5)
##                # get mirror fft
##                fft_ppg_mirror = getFFT(fft_ppg_input_data_mirror, sampling_rate_ppg)
##                # multiply by 60 conversion from bps -> bpm
##                freq_seconds_ppg_mirror = fft_ppg_mirror[1] * 60
##                # multiply by 60 conversion from bps -> bpm
##                freq_seconds_ppg_mirror = fft_ppg_mirror[1] * 60
####                print('fft_ppg_mirror.shape: ' + str(np.array(fft_ppg_mirror).shape))
##                # trim frequencies to get rid of very low bpm under 20, these are often large WE NEED DIFFERENT FILTER
##                freq_trim_index_ppg_mirror = np.where(freq_seconds_ppg_mirror == find_nearest(freq_seconds_ppg_mirror, 20))[0][0]
##                fft_freqs_ppg_mirror = np.asarray(fft_ppg_mirror[1])[freq_trim_index_ppg_mirror:]
##                fft_values_ppg_mirror = np.asarray(fft_ppg_mirror[0])[freq_trim_index_ppg_mirror:]
##                # find max value in fft_bpm_index_ppg_mirror
##                fft_bpm_index_ppg_mirror = np.where(fft_values_ppg_mirror == np.max(fft_values_ppg_mirror))[0][0]
##                fft_bpm_ppg_mirror = fft_freqs_ppg_mirror[fft_bpm_index_ppg_mirror]
##                self.fft_bpm_ppg = fft_bpm_ppg_mirror
####                print('self.fft_bpm_ppg: ' + str(self.fft_bpm_ppg))
####                print('max value index: ' + str(fft_bpm_index_ppg_mirror) + ', actual freq: ' + str(fft_bpm_ppg_mirror) + ', actual max value: ' + str(fft_values_ppg_mirror[fft_bpm_index_ppg_mirror]))
##                print('=== actual freq * 60: ' + str(np.round(fft_bpm_ppg_mirror, 3) * 60)
##                      + ', max value index: ' + str(np.round(fft_bpm_index_ppg_mirror, 3)) + ', actual freq: ' + str(np.round(fft_bpm_ppg_mirror, 3))
##                      + ', actual freq: ' + str(np.round(fft_bpm_ppg_mirror, 3)) + ', actual max value: ' + str(np.round(fft_values_ppg_mirror[fft_bpm_index_ppg_mirror], 3)))
##                print('=== actual freq * 60: ' + str(np.round(fft_bpm_ppg_mirror, 3) * 60)
##                      + ', fft_bpm_ppg_mirror_ir: ' + str(np.round((fft_bpm_ppg_mirror_ir * 60), 3))
##                      + ', fft_bpm_ppg_mirror_red: ' + str(np.round((fft_bpm_ppg_mirror_red * 60), 3)))
##                print('fft_bpm_ppg_mirror: ' + str(fft_bpm_ppg_mirror))
##                data_dict_bpm = {"heart_bpm": int(self.fft_bpm_ppg * 60)}


                # === PPG BPM MIRROR NEW 5/22/23
##                fft_ppg_input_data_ir = np.swapaxes(self.data_ppg, 0, 1)[1]
##                fft_ppg_input_data_red = np.swapaxes(self.data_ppg, 0, 1)[2]
####                fft_ppg_input_data_ir = np.swapaxes(self.plot_data_ppg, 0, 1)[1]
####                fft_ppg_input_data_red = np.swapaxes(self.plot_data_ppg, 0, 1)[2]
                # trim data to last 10 seconds
                fft_ppg_input_data_ir = np.swapaxes(self.data_ppg, 0, 1)[1]
                fft_ppg_input_data_red = np.swapaxes(self.data_ppg, 0, 1)[2]
                fft_ppg_input_data_ir = fft_ppg_input_data_ir[int(sampling_rate_ppg * 10):]
                fft_ppg_input_data_red = fft_ppg_input_data_red[int(sampling_rate_ppg * 10):]
                # mirror data +/- x5
                fft_ppg_input_data_mirror_ir = np.tile(np.append(fft_ppg_input_data_ir, np.flip(fft_ppg_input_data_ir, -1)), 5)
                fft_ppg_input_data_mirror_red = np.tile(np.append(fft_ppg_input_data_red, np.flip(fft_ppg_input_data_red, -1)), 5)
                # get mirror fft
                fft_ppg_mirror_ir = getFFT(fft_ppg_input_data_mirror_ir, sampling_rate_ppg)
                fft_ppg_mirror_red = getFFT(fft_ppg_input_data_mirror_red, sampling_rate_ppg)
                # multiply by 60 conversion from bps -> bpm
                freq_seconds_ppg_mirror_ir = fft_ppg_mirror_ir[1] * 60
                freq_seconds_ppg_mirror_red = fft_ppg_mirror_red[1] * 60
                # trim frequencies to get rid of very low bpm under 30, these are often large WE NEED DIFFERENT FILTER
##                freq_trim_index_ppg_mirror_ir = np.where(freq_seconds_ppg_mirror_ir == find_nearest(freq_seconds_ppg_mirror_ir, 20))[0][0]
                freq_trim_index_ppg_mirror_ir = np.where(freq_seconds_ppg_mirror_ir == find_nearest(freq_seconds_ppg_mirror_ir, 30))[0][0]
                fft_freqs_ppg_mirror_ir = np.asarray(fft_ppg_mirror_ir[1])[freq_trim_index_ppg_mirror_ir:]
                fft_values_ppg_mirror_ir = np.asarray(fft_ppg_mirror_ir[0])[freq_trim_index_ppg_mirror_ir:]
                # find max value in fft_bpm_index_ppg_mirror
                fft_bpm_index_ppg_mirror_ir = np.where(fft_values_ppg_mirror_ir == np.max(fft_values_ppg_mirror_ir))[0][0]
                fft_bpm_ppg_mirror_ir = fft_freqs_ppg_mirror_ir[fft_bpm_index_ppg_mirror_ir]
                # trim frequencies to get rid of very low bpm under 30, these are often large WE NEED DIFFERENT FILTER
##                freq_trim_index_ppg_mirror_red = np.where(freq_seconds_ppg_mirror_red == find_nearest(freq_seconds_ppg_mirror_red, 20))[0][0]
                freq_trim_index_ppg_mirror_red = np.where(freq_seconds_ppg_mirror_red == find_nearest(freq_seconds_ppg_mirror_red, 30))[0][0]
                fft_freqs_ppg_mirror_red = np.asarray(fft_ppg_mirror_red[1])[freq_trim_index_ppg_mirror_red:]
                fft_values_ppg_mirror_red = np.asarray(fft_ppg_mirror_red[0])[freq_trim_index_ppg_mirror_red:]
                # find max value in fft_bpm_index_ppg_mirror
                fft_bpm_index_ppg_mirror_red = np.where(fft_values_ppg_mirror_red == np.max(fft_values_ppg_mirror_red))[0][0]
                fft_bpm_ppg_mirror_red = fft_freqs_ppg_mirror_red[fft_bpm_index_ppg_mirror_red]
                # === get bpm: AVERAGE OR MAX (max helps control for noise in one channel while other gets good bpm
                fft_bpm_ppg_mirror = np.max([fft_bpm_ppg_mirror_ir, fft_bpm_ppg_mirror_red])
##                fft_bpm_ppg_mirror = (fft_bpm_ppg_mirror_ir + fft_bpm_ppg_mirror_red) / 2
                self.fft_bpm_ppg = fft_bpm_ppg_mirror
                # fill self.ppg_heart_rate_bpm_tracker
                self.ppg_heart_rate_bpm_tracker = np.roll(self.ppg_heart_rate_bpm_tracker, -1)
                self.ppg_heart_rate_bpm_tracker[len(self.ppg_heart_rate_bpm_tracker) - 1] = fft_bpm_ppg_mirror * 60
                # CHECK FOR MESSY DATA: HARD CUTOFF AT 40 BPM and if np.std(heart_rate_tracker) > 10
                if (fft_bpm_ppg_mirror * 60) < 40 or np.std(self.ppg_heart_rate_bpm_tracker) > 10:
                    data_dict_corner_widget = {"heart_bpm": "??", "eeg_artifact": "None", "accel_movement": "None", "gyro_movement": "None"}
                else:
                    data_dict_corner_widget = {"heart_bpm": int(self.fft_bpm_ppg * 60), "eeg_artifact": "None", "accel_movement": "None", "gyro_movement": "None"}
                # emit data
                self.new_data_corner_widget.emit(data_dict_corner_widget)

                # === PPG PSD
                if self.toggle_device_view == "PPG_PSD":
                    if self.current_filt_ppg == 0:
                        tf_data_buffer_ppg = (self.data_ppg - self.data_ppg.mean(axis=0))
                        tf_data_buffer_ppg = np.array([np.swapaxes(tf_data_buffer_ppg, 0, 1)])
                    else:
                        tf_data_buffer_ppg = np.array([np.swapaxes(self.data_ppg_filter_all[self.current_filt_ppg - 1], 0, 1)])
                    morlet_psd_ppg = tfr_array_morlet(tf_data_buffer_ppg, sfreq=sampling_rate_ppg, freqs=freqs_psd_ppg, output='power')[0]
                    for channel in range(0, len(morlet_psd_ppg)):
                        self.plot_data_psd_ppg[channel] = morlet_psd_ppg[channel] - np.mean(morlet_psd_ppg[channel])

##                    # === control for filter edge artifacts at begining of file
####                    range_max_buffer_ppg = len(self.data_ppg) - self.total_samples_ppg + (np.max(self.bf_ppg_index_offset) * 2)
##                    if self.current_filt_ppg != 0:
##                        if self.total_samples_ppg < len(self.data_ppg):
##                            range_max_buffer_ppg = len(self.data_ppg) - self.total_samples_ppg + (self.bf_ppg_index_offset[self.current_filt_ppg] * 2)
##                            if range_max_buffer_ppg <= len(self.data_ppg):
##                                for channel in range(0, len(morlet_psd_ppg)):
##                                    self.plot_data_psd_ppg[channel, :, 0:int(range_max_buffer_ppg)] = np.mean(morlet_psd_ppg[channel])
##                            else:
##                                for channel in range(0, len(morlet_psd_ppg)):
##                                    self.plot_data_psd_ppg[channel, :, :] = np.mean(morlet_psd_ppg[channel])

                    # update line/image data
                    self.image_data_ppg = self._update_image_data_psd_ppg()
##                    print('local_clock/timestamp_eeg/diff: ' + str(local_clock()) + ', ' + str(timestamps_eeg[len(timestamps_eeg) - 1]) + ', ' + str(local_clock() - timestamps_eeg[len(timestamps_eeg) - 1]))

                # === PPG FFT
                if self.toggle_device_view == "PPG_FFT":
                    data_ppg_fft = []
                    input_data_ppg_fft = np.swapaxes(self.plot_data_ppg, 0, 1)
                    for channel in range(0, n_chan_ppg_fft):
                        fft_data_buffer, fft_freq_buffer = getFFT(input_data_ppg_fft[channel][-window_ppg_fft_trim:], sampling_rate_ppg)
                        data_ppg_fft.append(fft_data_buffer)
                    data_ppg_fft, freq_ppg_fft = np.asarray(data_ppg_fft), fft_freq_buffer
                    self.data_ppg_fft = data_ppg_fft
##                    self.plot_data_ppg_fft = (np.swapaxes(self.data_ppg_fft, 0, 1))
                    self.plot_data_ppg_fft = (np.swapaxes(self.data_ppg_fft, 0, 1)) * 100
                    self._line_data_ppg_fft = self._update_line_data_ppg_fft()

                # === PPG TF
                if self.toggle_device_view in ['PPG_TF_Band', 'PPG_TF_BandChan']:
                    self.data_ppg_tf = np.roll(self.data_ppg_tf, -1, axis = 1)
                    input_data_ppg_tf = np.swapaxes(self.data_ppg[-window_ppg_tf_trim:], 0, 1)
                    bandpower = np.swapaxes(getBandpower_2023(input_data_ppg_tf, nperseg_ppg, nfft_ppg, sampling_rate_ppg, self.band_range_ppg), 0, 1)
                    self.data_ppg_tf[:, (len(self.data_ppg_tf[0]) - 1), :] = bandpower
                    self.data_ppg_tf_filter = np.roll(self.data_ppg_tf_filter, -1, axis = 2)
                    for filt in range(0, len(self.ppg_filter_list_range)):
                        input_data_ppg_tf_filt = np.swapaxes(self.data_ppg_filter_all[filt][-window_ppg_tf_trim:], 0, 1)
                        bandpower_filt = np.swapaxes(getBandpower_2023(input_data_ppg_tf_filt, nperseg_ppg, nfft_ppg, sampling_rate_ppg, self.band_range_ppg), 0, 1)
                        self.data_ppg_tf_filter[filt, :, (len(self.data_ppg_tf_filter[filt][0]) - 1), :] = bandpower_filt

                    # update data
##                    self.current_filt = self.filter_list_text_ppg.index(self.toggle_filt_ppg)
                    self.current_filt_ppg_buffer = self.filter_list_text_ppg.index(self.toggle_filt_ppg)
                    if self.current_filt_ppg == 0:
                        self.plot_data_ppg_tf_multiplot = self.data_ppg_tf
##                        self.plot_data_ppg_tf_multiplot = self.data_ppg_tf * 10
                    else:
                        self.plot_data_ppg_tf_multiplot = self.data_ppg_tf_filter[self.current_filt_ppg_buffer - 1]
##                        self.plot_data_ppg_tf_multiplot = self.data_ppg_tf_filter[self.current_filt_ppg - 1] * 10
                    # === TF Multi Plot: Separate Band, Separate Channel
                    if self.toggle_device_view == "PPG_TF_BandChan":
                        self._line_data_ppg_tf_multiplot = self._update_line_data_ppg_tf_multiplot()
                    # === TF Single Plot: Separate Band, Average Channel
                    if self.toggle_device_view == "PPG_TF_Band":
                        self.plot_data_ppg_tf_singleplot = np.swapaxes(self.plot_data_ppg_tf_multiplot.mean(axis = 2), 0, 1)
                        self._line_data_ppg_tf_singleplot = self._update_line_data_ppg_tf_singleplot()

                # === PPG ALL: EMIT DATA: All Device Line Data_Dict
                data_dict = {"toggle_view": self.toggle_device_view, "cam_toggle": self.cam_toggle, "line_toggles": self.line_bool_data_source, "play_bool": self.play_bool, "corner_widget_bool": self.corner_widget_bool,
                             "cam_toggle_count": self.cam_toggle_count,
                             "timestamps_len": len(timestamps_ppg), "artifact_pos_all": self.artifact_pos_all,"bf_eeg_index_offset": self.bf_eeg_index_offset,
                             "current_artifact_filt_eeg": self.filter_list_text_eeg.index(self.toggle_filt_eeg),
                             "EEG_PSD": self.image_data_eeg, "ACCEL_PSD": self.image_data_accel, "GYRO_PSD": self.image_data_gyro, "PPG_PSD": self.image_data_ppg,
                             "EEG_Time_Series": self._line_data_eeg, "ACCEL_Time_Series": self._line_data_accel, "GYRO_Time_Series": self._line_data_gyro, "PPG_Time_Series": self._line_data_ppg,
                             "EEG_FFT": self._line_data_eeg_fft, "ACCEL_FFT": self._line_data_accel_fft, "GYRO_FFT": self._line_data_gyro_fft, "PPG_FFT": self._line_data_ppg_fft,
                             "EEG_TF_Band": self._line_data_eeg_tf_singleplot, "ACCEL_TF_Band": self._line_data_accel_tf_singleplot, "GYRO_TF_Band": self._line_data_gyro_tf_singleplot, "PPG_TF_Band": self._line_data_ppg_tf_singleplot, 
                             "EEG_TF_BandChan": self._line_data_eeg_tf_multiplot, "ACCEL_TF_BandChan": self._line_data_accel_tf_multiplot, "GYRO_TF_BandChan": self._line_data_gyro_tf_multiplot, "PPG_TF_BandChan": self._line_data_ppg_tf_multiplot}
                self.new_data.emit(data_dict)

        # use QtCord.QTimer for updates
        QtCore.QTimer.singleShot(0, self.run_data_creation)

    # define update_line_data functions
    def _update_line_data_eeg(self):
        for channel in range(0, n_chan_eeg):
            self._line_data_eeg[channel, :, 1] = self.plot_data_eeg[:, channel]
        return self._line_data_eeg
    def _update_line_data_accel(self):
        for channel in range(0, n_chan_accel):
            self._line_data_accel[channel, :, 1] = self.plot_data_accel[:, channel]
        return self._line_data_accel
    def _update_line_data_gyro(self):
        for channel in range(0, n_chan_gyro):
            self._line_data_gyro[channel, :, 1] = self.plot_data_gyro[:, channel]
        return self._line_data_gyro
    def _update_line_data_ppg(self):
        for channel in range(0, n_chan_ppg):
            self._line_data_ppg[channel, :, 1] = self.plot_data_ppg[:, channel]
        return self._line_data_ppg
    def _update_line_data_eeg_fft(self):
        for channel in range(0, n_chan_eeg_fft):
            self._line_data_eeg_fft[channel, :, 1] = self.plot_data_eeg_fft[:, channel]
        return self._line_data_eeg_fft
    def _update_line_data_accel_fft(self):
        for channel in range(0, n_chan_accel_fft):
            self._line_data_accel_fft[channel, :, 1] = self.plot_data_accel_fft[:, channel]
        return self._line_data_accel_fft
    def _update_line_data_gyro_fft(self):
        for channel in range(0, n_chan_gyro_fft):
            self._line_data_gyro_fft[channel, :, 1] = self.plot_data_gyro_fft[:, channel]
        return self._line_data_gyro_fft
    def _update_line_data_ppg_fft(self):
        for channel in range(0, n_chan_ppg_fft):
            self._line_data_ppg_fft[channel, :, 1] = self.plot_data_ppg_fft[:, channel]
        return self._line_data_ppg_fft
    # === TF Single Plot: Separate Band, Average Channel
    def _update_line_data_eeg_tf_singleplot(self):
        for band in range(0, len(self.band_range_eeg)):
            self._line_data_eeg_tf_singleplot[band, :, 1] = self.plot_data_eeg_tf_singleplot[:, band]
        return self._line_data_eeg_tf_singleplot
    # === TF Multi Plot: Separate Band, Separate Channel
    def _update_line_data_eeg_tf_multiplot(self):
        for channel in range(0, n_chan_eeg_tf):
            for band in range(0, len(self.band_range_eeg)):
##                line_index_buffer = int((band + (n_chan_eeg_tf * channel)))
                line_index_buffer = int((band + (len(self.band_range_eeg) * channel)))
##                self._line_data_eeg_tf_multiplot[channel, :, 1] = self.plot_data_eeg_tf_multiplot[band, :, channel]
                self._line_data_eeg_tf_multiplot[line_index_buffer, :, 1] = self.plot_data_eeg_tf_multiplot[band, :, channel]
        return self._line_data_eeg_tf_multiplot

    # === TF Single Plot: Separate Band, Average Channel
    def _update_line_data_accel_tf_singleplot(self):
        for band in range(0, len(self.band_range_accel)):
            self._line_data_accel_tf_singleplot[band, :, 1] = self.plot_data_accel_tf_singleplot[:, band]
        return self._line_data_accel_tf_singleplot
    # === TF Multi Plot: Separate Band, Separate Channel
    def _update_line_data_accel_tf_multiplot(self):
        for channel in range(0, n_chan_accel_tf):
            for band in range(0, len(self.band_range_accel)):
                line_index_buffer = int((band + (len(self.band_range_accel) * channel)))
                self._line_data_accel_tf_multiplot[line_index_buffer, :, 1] = self.plot_data_accel_tf_multiplot[band, :, channel]
        return self._line_data_accel_tf_multiplot
    # === TF Single Plot: Separate Band, Average Channel
    def _update_line_data_gyro_tf_singleplot(self):
        for band in range(0, len(self.band_range_gyro)):
            self._line_data_gyro_tf_singleplot[band, :, 1] = self.plot_data_gyro_tf_singleplot[:, band]
        return self._line_data_gyro_tf_singleplot
    # === TF Multi Plot: Separate Band, Separate Channel
    def _update_line_data_gyro_tf_multiplot(self):
        for channel in range(0, n_chan_gyro_tf):
            for band in range(0, len(self.band_range_gyro)):
                line_index_buffer = int((band + (len(self.band_range_gyro) * channel)))
                self._line_data_gyro_tf_multiplot[line_index_buffer, :, 1] = self.plot_data_gyro_tf_multiplot[band, :, channel]
        return self._line_data_gyro_tf_multiplot
    # === TF Single Plot: Separate Band, Average Channel
    def _update_line_data_ppg_tf_singleplot(self):
        for band in range(0, len(self.band_range_ppg)):
            self._line_data_ppg_tf_singleplot[band, :, 1] = self.plot_data_ppg_tf_singleplot[:, band]
        return self._line_data_ppg_tf_singleplot
    # === TF Multi Plot: Separate Band, Separate Channel
    def _update_line_data_ppg_tf_multiplot(self):
        for channel in range(0, n_chan_ppg_tf):
            for band in range(0, len(self.band_range_ppg)):
                line_index_buffer = int((band + (len(self.band_range_ppg) * channel)))
                self._line_data_ppg_tf_multiplot[line_index_buffer, :, 1] = self.plot_data_ppg_tf_multiplot[band, :, channel]
        return self._line_data_ppg_tf_multiplot

    # === PSD Multi Plot: Separate Band, Separate Channel
##    def _update_image_data_psd(self):
##        self._image_data = self.plot_data_psd
##        return self._image_data.copy()

    def _update_image_data_psd_eeg(self):
        self._image_data = self.plot_data_psd_eeg
        return self._image_data.copy()
    def _update_image_data_psd_accel(self):
        self._image_data = self.plot_data_psd_accel
        return self._image_data.copy()
    def _update_image_data_psd_gyro(self):
        self._image_data = self.plot_data_psd_gyro
        return self._image_data.copy()
    def _update_image_data_psd_ppg(self):
        self._image_data = self.plot_data_psd_ppg
        return self._image_data.copy()

    # define update_toggles
    def update_toggles(self, new_data_dict_toggle):
        self.cam_toggle_count = new_data_dict_toggle["cam_toggle_count"]
        self.corner_widget_bool, self.line_bool_data_source = new_data_dict_toggle["corner_widget_bool"], new_data_dict_toggle["line_toggles"]
        self.cam_toggle, self.toggle_device_view, self.play_bool = new_data_dict_toggle["cam_toggle"], new_data_dict_toggle["view_toggle"], new_data_dict_toggle["play_bool"]
        self.toggle_filt_eeg, self.toggle_filt_accel, self.toggle_filt_gyro, self.toggle_filt_ppg = new_data_dict_toggle["filt_toggle_eeg"], new_data_dict_toggle["filt_toggle_accel"], new_data_dict_toggle["filt_toggle_gyro"], new_data_dict_toggle["filt_toggle_ppg"]
##        self.play_bool = new_data_dict_toggle["play_bool"]
##        self.cam_toggle = new_data_dict_toggle["cam_toggle"]
##        self.toggle_device_view = new_data_dict_toggle["view_toggle"]
##        self.line_bool_data_source = new_data_dict_toggle["line_toggles"]
##        self.corner_widget_bool = new_data_dict_toggle["corner_widget_bool"]
##        self.toggle_filt_eeg = new_data_dict_toggle["filt_toggle_eeg"]
##        self.toggle_filt_accel, self.toggle_filt_gyro, self.toggle_filt_ppg = new_data_dict_toggle["filt_toggle_accel"], new_data_dict_toggle["filt_toggle_gyro"], new_data_dict_toggle["filt_toggle_ppg"]
##        print('self.toggle_view: ' + str(self.toggle_device_view) + ', self.toggle_filt_eeg: ' + str(self.toggle_filt_eeg) + ', self.cam_toggle: ' + str(self.cam_toggle)
##              + ', self.toggle_filt_accel: ' + str(self.toggle_filt_accel) + ', self.toggle_filt_gyro: ' + str(self.toggle_filt_gyro))

    def stop_data(self):
        print("Data source is quitting...")
        self._should_end = True

if __name__ == "__main__":
    app = use_app("pyqt5")
    app.create()

    canvas_wrapper = CanvasWrapper()
    win = MyMainWindow(canvas_wrapper)
    data_thread = QtCore.QThread(parent=win)
    data_source = DataSource()
    data_source.moveToThread(data_thread)

    # update the visualization when there is new data
####    win.new_data_filt.connect(canvas_wrapper.update_toggles)                # OPTIONAL PATHWAY: No need for it yet but it is direct connect: MyMainWindow --> Canvas
##    win.new_data_line_toggles.connect(canvas_wrapper.update_toggles)          # OPTIONAL PATHWAY: No need for it yet but it is direct connect: MyMainWindow --> Canvas

    win.new_data_toggle.connect(data_source.update_toggles)
    data_source.new_data.connect(canvas_wrapper.update_data)
##    data_source.new_data_bpm.connect(win.update_bpm)
    data_source.new_data_corner_widget.connect(win.update_corner_widget)

##    data_source.new_data_axes_labels.connect(win.update_axes_labels)

    # start data generation when the thread is started
    data_thread.started.connect(data_source.run_data_creation)
    # if the data source finishes before the window is closed, kill the thread
    # to clean up resources
    data_source.finished.connect(data_thread.quit)
    # if the window is closed, tell the data source to stop
    win.closing.connect(data_source.stop_data)
    # when the thread has ended, delete the data source from memory
    data_thread.finished.connect(data_source.deleteLater)

    win.show()
    data_thread.start()
    app.run()

    print("Waiting for data source to close gracefully...")
    data_thread.quit()
    data_thread.wait(5000)

