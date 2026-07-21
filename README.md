# Melodict 🎹📈

**Real-time symbolic melody extraction, phrase segmentation, and dictionary generation for interactive music systems.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-purple.svg)](https://github.com/astral-sh/uv)

## Overview

**Melodict** bridges the gap between raw audio capture (via Max/MSP or live instruments like piano and guitar) and symbolic AI generation. Designed to overcome traditional FFT filtering delays and legacy pitch-tracking bottlenecks (e.g., Skyline algorithm), Melodict provides an ultra-low-latency Open Sound Control (OSC) pipeline that:

1. **Extracts** symbolic pitch and MIDI events in real-time using State-of-the-Art (SOTA) lightweight neural networks and acoustic feature trackers.
2. **Segments** continuous melodic streams into natural musical phrases using Gestalt-based heuristic models (Local Boundary Detection Model - LBDM).
3. **Builds** structured phrase dictionaries on the fly, enabling reactive AI accompaniment and continuation models (e.g., Variable-order Markov Models / Factor Oracles).

## Architecture

    [ Live Piano / Guitar ] ---> [ Max/MSP Capture ] --(UDP / OSC < 2ms)--> [ Melodict Python Engine ]
                                                                                       │
    [ Reactive AI Accompaniment ] <-- [ Phrase Dictionary ] <-- [ LBDM Segmentation ] <┘

## Quick Start

### 1. Prerequisites
* uv (https://github.com/astral-sh/uv - for local Python dependency management)
* Docker & Docker Compose (optional, for containerized execution)
* Cycling '74 Max/MSP (for live audio capture and synthesis)

### 2. Local Setup with uv

Clone the repository and sync dependencies instantly:

    git clone https://github.com/yourusername/melodict.git
    cd melodict
    uv sync

Run the OSC bridge server locally:

    uv run python -m melodict.osc.server --ip 127.0.0.1 --port 8001

### 3. Containerized Setup (Docker)

To run the engine in an isolated environment with host networking (zero UDP port-forwarding latency):

    docker compose up --build

## Testing with Datasets (Piano & Guitar)

To benchmark pitch-tracking accuracy and segmentation boundaries against academic ground-truths, use our automated downloader:

    uv run python scripts/download_datasets.py --target all

Run an offline segmentation evaluation:

    uv run python scripts/simulate_live_stream.py --dataset wjazzd --speed 1.0

## Max/MSP Integration

1. Open max_msp/osc_bridge.maxpat in Max 8 or later.
2. Connect your audio interface input (microphone or guitar instrument cable) to the adc~ object.
3. Ensure the UDP send port is set to 8001 and receive port is set to 8000.
4. Turn on DSP to begin streaming audio features and receiving segmented symbolic phrases.

## License

This project is licensed under the MIT License - see the LICENSE file for details.