# Melodict

**Real-time symbolic melody extraction, phrase segmentation, and dictionary generation for interactive music systems.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-purple.svg)](https://github.com/astral-sh/uv)

## Overview

**Melodict** bridges the gap between raw audio capture (via Max/MSP or live instruments like piano and guitar) and symbolic AI generation. Designed to overcome traditional FFT filtering delays and legacy pitch-tracking bottlenecks, Melodict provides a hybrid zero-latency pipeline using **BlackHole** (CoreAudio) for PCM streaming and **Open Sound Control (OSC)** for symbolic control data:

1. **Extracts** symbolic pitch and MIDI events in real-time using State-of-the-Art (SOTA) neural networks and acoustic trackers.
2. **Segments** continuous melodic streams into natural musical phrases using Gestalt-based heuristic models (LBDM).
3. **Builds** structured phrase dictionaries on the fly, enabling reactive AI accompaniment.

### Phase 1: Real-Time Melody Extraction Breakthroughs
During our initial evaluation phase on the MAESTRO (Piano) and GuitarSet datasets, we implemented two critical architectural improvements:
* **Smart Skyline Ground Truth:** We enhanced the traditional Skyline algorithm with offline contextual windowing to detect and filter out "bass bleeding" during melodic rests, providing a mathematically pure Ground Truth for evaluations.
* **Asymmetric Sliding Window:** To solve the latency vs. context dilemma, we wrapped Convolutional Neural Networks (like Spotify's `basic-pitch`) in a historical circular buffer. The audio advances in ultra-fast 46 ms increments, allowing the CNN to utilize deep polyphonic context while delivering zero-perceived-latency updates to the musician.

![Melody Extraction Visualization](assets/paper_plot.png)
*Figure 1: Frame-level visualization (46ms buffer) comparing SOTA engines against the MAESTRO dataset. The Smart Skyline (Solid Black) successfully ignores the bass-bleeding errors (Dotted Red) that pollute traditional extraction. The 2.0s buffered basic-pitch (Purple) successfully tracks the true melody through polyphonic noise where acoustic algorithms fail.*

**Key Findings (Empirical Benchmarks):**
* **Latency:** Acoustic trackers like `essentia-yin` process 46ms frames in just **3.5 ms**. Neural Networks require the circular buffer architecture to function in live environments.
* **Complex Polyphony (Piano):** The CNN `basic-pitch` dominates on the MAESTRO dataset (0.39 RPA vs 0.05 RPA for acoustic models) due to its ability to filter harmonic bleeding and room reverberation.
* **Clean Monophonic (Guitar):** When processing clean, line-in guitar signals (GuitarSet), the acoustic `essentia-yin` engine slightly outperforms the CNN (0.447 RPA vs 0.445 RPA), proving acoustic algorithms are optimal for direct-input instruments.
* **Interactive Generation:** The VMM Continuator processes musical context and generates stylistic responses in **< 0.02 ms**, guaranteeing seamless human-machine interaction.

### Phase 1.5: DSP Engine Temporal & Acoustic Benchmarking
To measure real-world reliability, engines were subjected to a 5-second per note chromatic scale test (A2 to A5). We evaluated them under both **Clean Audio** conditions and simulated **Live Stage Acoustics** (50Hz rumble, ambient noise, room reverb, and harmonic complexity).

**Table 1: Clean Audio Simulation**
| Engine | Accuracy (%) | Octave Errors (%) | Missed Voicing (%) | False Alarms (%) | Lag (ms) |
|---|---|---|---|---|---|
| **librosa-pyin** | 99.06 | 0.00 | 0.00 | 0.94 | 0.0 |
| **essentia-yin** | 99.06 | 0.00 | 0.00 | 0.94 | 0.0 |
| **basic-pitch** | 96.87 | 0.00 | 2.29 | 0.85 | 106.9 |
| **crepe-tiny** | 90.58 | 0.00 | 8.23 | 0.72 | 51.0 |

![DSP Benchmark Clean](assets/dsp_benchmark_clean.png)

**Table 2: Live Stage Acoustics Simulation**
| Engine | Accuracy (%) | Octave Errors (%) | Missed Voicing (%) | False Alarms (%) | Lag (ms) |
|---|---|---|---|---|---|
| **basic-pitch** | 96.79 | 0.00 | 1.17 | 0.92 | 108.2 |
| **librosa-pyin** | 96.49 | 2.56 | 0.00 | 0.94 | 0.0 |
| **essentia-yin** | 93.96 | 5.12 | 0.00 | 0.92 | 0.0 |
| **crepe-tiny** | 81.28 | 0.12 | 17.88 | 0.67 | 19.9 |

![DSP Benchmark Acoustic](assets/dsp_benchmark_acoustic.png)
*Figure 2: Performance degradation under stage acoustics. While `essentia-yin` and `librosa-pyin` suffer from mathematically induced octave errors (Red 'x') when confronted with harmonic resonance, the neural engine `basic-pitch` maintains perfect octave stability at the cost of a higher transition latency.*

**Note on Temporal Precision & Ground Truth Accuracy:** To prevent digital audio clipping during instantaneous pitch transitions, the benchmark applies a 5ms ADSR fade-in/fade-out envelope to the edges of every note. The Ground Truth array strictly models these 5ms windows as genuine silences (MIDI 0). 

Consequently, neural engines like `basic-pitch` possess the temporal resolution to detect these millisecond drops. In contrast, acoustic engines that rely on Viterbi decoding or temporal smoothing (like `librosa-pyin`) often gloss over these micro-rests, artificially bridging the gap and incurring accuracy penalties (False Alarms) for hallucinating a note where silence physically exists.

### Phase 2: Persistent Corpus & LBDM Segmentation

Melodict builds a persistent musical memory by extracting performances, applying quality filters (bass-bleeding removal, median filtering), and segmenting them using Gestalt principles and the Local Boundary Detection Model (LBDM). The resulting high-quality phrases are stored in a PostgreSQL relational database.

**1. Audio Extraction (Neural Network + State Machine)**
*Extracting polyphonic audio via Basic-Pitch with a 46ms sliding window and Gestalt cutoffs.*
![Corpus Audit Audio](assets/corpus_audit_MAESTRO_Generic_Audio.png)

**2. Symbolic Extraction (Smart Skyline on MIDI)**
*Direct symbolic processing using the Smart Skyline algorithm to extract the highest pitch in real-time, yielding pristine musical phrases.*
![Corpus Audit MIDI](assets/corpus_audit_MAESTRO_Generic_Midi.png)

## Architecture

    [ Live Piano / Guitar ] ---> [ Max/MSP (Out: BlackHole) ] ==(CoreAudio PCM: 0ms)==> [ Melodict Python ]
                                                                                                │
    [ Reactive AI Accompaniment ] <--(UDP / OSC: < 2ms)-- [ Phrase Dictionary ] <-- [ LBDM ] <──┘

## Quick Start

### 1. Prerequisites
* **uv**: An extremely fast Python package installer and resolver written in Rust. It replaces `pip` and `virtualenv`, ensuring reproducible and instantaneous environment setups. (https://github.com/astral-sh/uv)
* **BlackHole**: A virtual audio driver for zero-latency routing on macOS.
* **PostgreSQL**: For the persistent musical phrase corpus.
* **Cycling '74 Max/MSP**: For live audio capture and synthesis.

### 2. Local Setup with uv

Clone the repository and sync all dependencies instantly:

    git clone https://github.com/yourusername/melodict.git
    cd melodict
    uv sync

### 3. Running a Live Interactive Session

The core orchestrator (`scripts/live_session.py`) connects BlackHole audio with the PostgreSQL corpus and Max/MSP OSC outputs. 

**Free Improvisation (Load full DB):**
    
    uv run python scripts/live_session.py

**Stylistic Filter (Only load specific artists/sessions into Markov memory):**
    
    uv run python scripts/live_session.py --filter-artists "Ignasi, Bill Evans" --filter-sessions "Session_1"

**Active Learning (Save your live performance phrases into the database):**

    uv run python scripts/live_session.py --learn-as "Philippe_Live_1"

## Max/MSP Integration

1. Open `max_msp/osc_bridge.maxpat` in Max 8 or later.
2. Go to **Options -> Audio Status** and set your **Output Device** to `BlackHole`.
3. Connect your hardware audio input (microphone or instrument cable) to the `ezadc~` object.
4. Turn on the DSP engine. Audio is now silently routed to Python with zero latency.
5. Max receives the AI-generated MIDI responses dynamically via OSC on UDP port `8000`.

## Testing & Evaluation Scripts

To benchmark pitch-tracking accuracy against academic ground-truths, use our automated tools:

Download datasets (MAESTRO & GuitarSet):

    uv run python scripts/download_datasets.py --target all --mode full

Run the Grid Search to find the optimal Latency/Context ratio for Neural Networks:

    uv run python scripts/optimize_basic_pitch.py --data_dir /path/to/maestro --max_files 10 --duration 15.0

Visualize predictions vs. Ground Truth on a Piano Roll:

    uv run python scripts/visualize_melody.py --audio_file /path/to/audio.wav --buffer_ms 46 --context_sec 2.0

Evaluate algorithms specifically on Guitar audio (Acoustic Mic vs. Line-In Hexaphonic):

    uv run python scripts/evaluate_guitarset.py --data_dir /path/to/guitarset --audio_type mix
    
Run DSP & Temporal Precision benchmarks:

    uv run python scripts/benchmark_dsp_analysis.py --note_duration 5.0

## License

**All rights reserved.** This software and associated documentation files are proprietary and intended solely for academic research and evaluation purposes. No public licensing, commercial reuse, modification, or unauthorized distribution is permitted without explicit written consent from the author.