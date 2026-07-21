# Max/MSP OSC Bridge 🎛️

This directory contains the Cycling '74 Max/MSP patches required to capture live audio from a piano or guitar and communicate with the **Melodict** Python engine with ultra-low latency.

## Prerequisites
* **Max 8** or newer.
* (Optional) **Aubio external objects for Max** (`aubiopitch~`, `aubioonset~`) installed in your Max Packages folder for C++ native acoustic pitch tracking.

## Setup Instructions

1. Launch `osc_bridge.maxpat` in Max/MSP.
2. Verify the **UDP networking parameters**:
   * `udpsend`: Target IP `127.0.0.1`, Port `8001` (Sends features to Python).
   * `udpreceive`: Port `8000` (Receives predicted MIDI notes and phrase boundaries from Python).
3. Connect your hardware audio input (microphone for piano, audio interface for electric guitar) to the `adc~` input box.
4. Click the **eZbac~ (Speaker icon)** to turn on DSP processing.

## OSC Data Formatting

* **Outgoing (Max -> Python):**
  * `/audio/features <float: frequency>`: Sends detected fundamental frequency in Hz.
  * `/midi/note_in <int: pitch> <int: velocity>`: Sends live MIDI keyboard events.
* **Incoming (Python -> Max):**
  * `/midi/note_out <int: pitch>`: Real-time cleaned symbolic pitch stream.
  * `/phrase/boundary <int: trigger>`: Bang trigger when LBDM detects the end of a musical phrase.