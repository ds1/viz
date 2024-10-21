# Petal Viz 1.0
## Visualizer for data streamed from Petal Metrics and the Petal APIs.

# Project Requirements and Instructions
Transform a proof of concept python file into a refactored real-time data streaming visualization tool based on an updated interface mockup.

This new app, let's call Viz, displays a time-series visualization of an incoming data stream from a separate app, Petal Metrics, over the local network via Lab Streaming Layer (LSL).

The stream sends EEG, Gyroscope, Accelerometer , Telemetry, and PPG data.

The data in the graph should move smoothly from right to left. If no data from the stream is available, the graph should pause movement and the status bar should be updated appropriately.

The user can switch views to display one of these types of data at a time.

The user can select to apply one of the various filters to the signal data.

The user should be able to zoom in and out to make the data stream appear larger or smaller, respectively (Vertical Scale).

The graph should have a timeline beneath it that aligns with the graph so that the last time window of x.xxxx (e.g., 4.000) seconds of data are shown (e.g., 4 seconds ago through now).

The user should be able to change the time window from pre-defined options (e.g., 2 seconds, 4 seconds (default), and 8 seconds).

The graph should be able to display various numbers of channels (e.g., 4 for EEG: Left Ear, TP9; Left Forehead, FP1; Right Forehead, FP2; and Right Ear, TP10;).

The graph should be able to be viewed in monochrome (data from all channels are the same color) or multicolor (data from each channel has a unique color).

At the top of the app window, there should be a bar that contains:
- App icon
- App name (Viz) and version (0.1)
- Menu dropdowns (View: Zoom In, Zoom Out, Monochrome, Multicolor; Window: Enter Fullscreen, Exit Fullscreen; Help: User Guide (links to https://docs.petal.tech), Support (links to https://docs.petal.tech);

At the bottom of the app window, there should be a status bar that displays the following:
- Console log (limited to one-line (256 characters) of text
- Stream status (Disconnected, Discovering, Connecting, Trying to Reconnect, Streaming)
- Output settings (Stream name, Stream type, Port, Destination IP)
- Input device status (Device name & ID (e.g, Muse-ADB3)
- Bluetooth icon that differs depending on the stream status
- Battery icon and battery life percentage

Reference the Petal Material Design Style Guide for colors, typography, components, and icons. Additional components needed should use components from Material Design according to the Design System.

Eventually, more panes will be added to the interface as the number of features grows. The focus of this first release will be to display the 'Visualizer' ,'Timeline', and 'Status bar' panes. Future panes include 'Settings' and 'Markers'.

Recommend the best approach for the evolution of the product from the simple proof of concept to a high-performant desktop application.

If the app can support the high graphical performance needs, using Electron and web technologies like CSS is ok. Otherwise, recommend the best options.

Your job, as an expert software engineer, is to develop the architecture, code, and processes for the product to the best of your ability.

# Reference Materials
## Design Assets
### Original proof of concept prototype: 
vispyqt_visualizer_MVP_v8.py
### Complementary files for the proof of concept prototype:
vispyqt_visualizer_helperFuncsTest.py
vispyqt_visualizer_helperFuncs_v25.py

### Interface mockup of this app we are aiming to build in this iteration: Petal_Viz_1.0_Mockup.pdf

### Reference for future iterations of the app (out of scope for this project): Petal_Viz_Future_Mockup.pdf