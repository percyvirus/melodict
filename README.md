# Melodict

**Real-time symbolic melody extraction, phrase segmentation, and dictionary generation for interactive music systems.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-purple.svg)](https://github.com/astral-sh/uv)

## Overview

**Melodict** bridges the gap between raw audio capture (via Max/MSP or live instruments like piano and guitar) and symbolic AI generation. Designed to overcome traditional FFT filtering delays and legacy pitch-tracking bottlenecks, Melodict provides an ultra-low-latency Open Sound Control (OSC) pipeline that:

1. **Extracts** symbolic pitch and MIDI events in real-time using State-of-the-Art (SOTA) neural networks and acoustic trackers.
2. **Segments** continuous melodic streams into natural musical phrases using Gestalt-based heuristic models (LBDM).
3. **Builds** structured phrase dictionaries on the fly, enabling reactive AI accompaniment.

### Phase 1: Real-Time Melody Extraction Breakthroughs
During our initial evaluation phase on the MAESTRO (Piano) and GuitarSet datasets, we implemented two critical architectural improvements:
* **Smart Skyline Ground Truth:** We enhanced the traditional Skyline algorithm with offline contextual windowing to detect and filter out "bass bleeding" during melodic rests, providing a mathematically pure Ground Truth for evaluations.
* **Asymmetric Sliding Window:** To solve the latency vs. context dilemma, we wrapped Convolutional Neural Networks (like Spotify's `basic-pitch`) in a 2.0-second historical circular buffer. The audio advances in ultra-fast 46 ms increments, allowing the CNN to utilize deep polyphonic context while delivering zero-perceived-latency updates to the musician.

## Architecture

    [ Live Piano / Guitar ] ---> [ Max/MSP Capture ] --(UDP / OSC < 2ms)--> [ Melodict Python Engine ]
                                                                                       │
    [ Reactive AI Accompaniment ] <-- [ Phrase Dictionary ] <-- [ LBDM Segmentation ] <┘

## Quick Start

### 1. Prerequisites
* uv ([https://github.com/astral-sh/uv](https://github.com/astral-sh/uv) - for local Python dependency management)
* Docker & Docker Compose (optional, for containerized execution)
* Cycling '74 Max/MSP (for live audio capture and synthesis)

### 2. Local Setup with uv

Clone the repository and sync dependencies instantly:

    git clone [https://github.com/yourusername/melodict.git](https://github.com/yourusername/melodict.git)
    cd melodict
    uv sync

Run the OSC bridge server locally:

    uv run python -m melodict.osc.server --ip 127.0.0.1 --port 8001

### 3. Containerized Setup (Docker)

To run the engine in an isolated environment with host networking (zero UDP port-forwarding latency):

    docker compose up --build

## Testing & Evaluation Scripts

To benchmark pitch-tracking accuracy against academic ground-truths, use our automated tools:

Download datasets (MAESTRO & GuitarSet):

    uv run python scripts/download_datasets.py --target all --mode full

Run the Grid Search to find the optimal Latency/Context ratio for Neural Networks:

    uv run python scripts/optimize_basic_pitch.py --data_dir /path/to/maestro --max_files 10 --duration 15.0

Visualize predictions vs. Ground Truth on a Piano Roll:

    uv run python scripts/visualize_melody.py --audio_file /path/to/audio.wav --buffer_ms 46 --context_sec 2.0

Evaluate algorithms specifically on Guitar audio (Acoustic Mic vs. Line-In Hexaphonic):

    uv run python scripts/evaluate_guitarset.py --data_dir /path/to/guitarset --audio_type mic

## Max/MSP Integration

1. Open `max_msp/melodict_osc_bridge.maxpat` in Max 8 or later.
2. Connect your audio interface input (microphone or guitar instrument cable) to the `ezadc~` object.
3. Ensure the UDP send port is set to `8001` and receive port is set to `8000`.
4. Turn on DSP to begin streaming audio features.

## License

**All rights reserved.** This software and associated documentation files are proprietary and intended solely for academic research and evaluation purposes. No public licensing, commercial reuse, modification, or unauthorized distribution is permitted without explicit written consent from the author.