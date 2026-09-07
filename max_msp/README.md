Max/MSP & BlackHole Integration

This directory contains the Max/MSP patches required to capture live audio and communicate with the Melodict Python engine using zero-latency CoreAudio routing.

Setup Instructions

1. Launch osc_bridge.maxpat or melodict_standalone.maxpat in Max/MSP.
2. Go to Options -> Audio Status in Max, and set your Output Device to BlackHole.
3. Connect your hardware audio input to the ezadc~ input box.
4. Click the ezadc~ to turn on DSP processing. The audio is now silently routed to Python.
5. Ensure Python is listening via: uv run python scripts/blackhole_live_capture.py --buffer_ms 46
6. Max still receives AI responses via OSC on udpreceive port 8000.