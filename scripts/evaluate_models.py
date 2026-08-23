"""
Audio-to-MIDI Melody Evaluation Pipeline for Melodict.
Benchmarks SOTA real-time engines using Frame-Level Predominant Melody Extraction metrics.
"""

import argparse
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import polars as pl
import pretty_midi
from scipy.signal import medfilt
from tqdm import tqdm

from melodict.extraction.sota_models import EngineFactory, PitchExtractorEngine


def predict_with_realtime_engine_melody(
    engine: PitchExtractorEngine, 
    audio_path: Path, 
    max_duration: float = 15.0, 
    buffer_ms: int = 46
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates real-time processing frame-by-frame and applies temporal smoothing 
    to output continuous pitch trajectories instead of strict intervals.
    """
    sample_rate = 44100
    hop_length = int(sample_rate * (buffer_ms / 1000.0))
    
    # Load audio (mono) up to max_duration
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=max_duration)
    
    est_times = []
    est_freqs = []
    
    # Simulate a live streaming buffer
    for i in range(0, len(y), hop_length):
        buffer = y[i : i + hop_length]
        if len(buffer) < hop_length:
            break  
            
        current_time_sec = i / sample_rate
        predicted_midi = engine.predict_frame(buffer, sample_rate)
        
        est_times.append(current_time_sec)
        
        if predicted_midi is not None:
            est_freqs.append(librosa.midi_to_hz(predicted_midi))
        else:
            est_freqs.append(0.0) # 0.0 indicates unvoiced/silence in mir_eval
            
    est_times = np.array(est_times)
    est_freqs = np.array(est_freqs)
    
    # --- TEMPORAL SMOOTHING (HYSTERESIS) ---
    # Apply a median filter of 5 frames (approx 230ms at 46ms hop)
    # This removes 1-frame jitters and bridges tiny gaps caused by acoustic polyphonic interference
    if len(est_freqs) >= 5:
        est_freqs = medfilt(est_freqs, kernel_size=5)
        
    return est_times, est_freqs


def load_ground_truth_skyline_melody(midi_path: Path, max_duration: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses a Ground Truth MIDI file and generates a continuous time-frequency 
    trajectory using the Skyline algorithm (highest pitch tracking).
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    
    dt = 0.01  # 10ms high-resolution time grid
    time_grid = np.arange(0, max_duration, dt)
    skyline_pitches = np.zeros_like(time_grid)
    
    # Apply Skyline Algorithm
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                if note.start >= max_duration:
                    continue
                end_time = min(note.end, max_duration)
                
                start_idx = int(note.start / dt)
                end_idx = int(end_time / dt)
                
                if start_idx < len(skyline_pitches):
                    # Keep only the highest pitch in this time window
                    skyline_pitches[start_idx:end_idx] = np.maximum(
                        skyline_pitches[start_idx:end_idx], note.pitch
                    )
                    
    ref_times = time_grid
    ref_freqs = np.zeros_like(skyline_pitches)
    
    # Convert active MIDI pitches to Hz (0 remains 0 for unvoiced)
    active_idx = skyline_pitches > 0
    ref_freqs[active_idx] = librosa.midi_to_hz(skyline_pitches[active_idx])
    
    return ref_times, ref_freqs


def evaluate_dataset(dataset_dir: Path, max_files: int = 3, max_duration: float = 15.0, buffer_ms: int = 46) -> pl.DataFrame:
    """
    Iterates through dataset audio-midi pairs, runs all engines, and calculates frame-level metrics.
    """
    engines = EngineFactory.get_all_available()
    if not engines:
        print("No pitch extraction engines available to benchmark.")
        return pl.DataFrame()
        
    print(f"\nLoaded {len(engines)} real-time engines for benchmarking:")
    for e in engines:
        print(f"  - {e.name}")

    results = []
    
    audio_files = list(dataset_dir.rglob("*.wav"))[:max_files]
    print(f"\nEvaluating {len(audio_files)} audio files (First {max_duration}s per file @ {buffer_ms}ms buffer)...\n")
    
    for audio_path in tqdm(audio_files, desc="Dataset Progress", position=0):
        midi_path = audio_path.with_suffix(".midi")
        if not midi_path.exists():
            midi_path = audio_path.with_suffix(".mid")
        if not midi_path.exists():
            continue
            
        # Load continuous Ground Truth
        ref_times, ref_freqs = load_ground_truth_skyline_melody(midi_path, max_duration)
        if len(ref_times) == 0:
            continue
            
        for engine in engines:
            engine.reset_context()
            
            est_times, est_freqs = predict_with_realtime_engine_melody(engine, audio_path, max_duration, buffer_ms)
            
            if len(est_times) == 0:
                rpa, vr, vfa = 0.0, 0.0, 0.0
            else:
                # Calculate MIR Melody metrics (Raw Pitch Accuracy, Voicing Recall, Voicing False Alarm)
                metrics = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
                rpa = metrics.get('Raw Pitch Accuracy', 0.0)
                vr = metrics.get('Voicing Recall', 0.0)
                vfa = metrics.get('Voicing False Alarm', 0.0)
                
            results.append({
                "engine": engine.name,
                "file_name": audio_path.name,
                "raw_pitch_acc": round(float(rpa), 4),
                "voicing_recall": round(float(vr), 4),
                "false_alarm": round(float(vfa), 4)
            })
            
    return pl.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Melodict Engines on Predominant Melody Extraction.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True,
        help="Path to the extracted dataset folder (e.g., /Volumes/PERCYVIRUS/DATASETS/maestro/maestro-v3.0.0)"
    )
    parser.add_argument(
        "--max_files", 
        type=int, 
        default=3,
        help="Maximum number of files to process."
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        default=10.0,
        help="Seconds of audio to process per file (default: 10.0s)."
    )
    parser.add_argument(
        "--buffer_ms", 
        type=int, 
        default=46,
        help="Buffer size in ms to simulate live tracking latency (default: 46)."
    )
    
    args = parser.parse_args()
    data_path = Path(args.data_dir)
    
    if not data_path.exists():
        print(f"Error: Dataset directory not found at {data_path}")
        return
        
    df_metrics = evaluate_dataset(
        data_path, 
        max_files=args.max_files, 
        max_duration=args.duration,
        buffer_ms=args.buffer_ms
    )
    
    if len(df_metrics) > 0:
        print("\n=== Granular Evaluation Results ===")
        print(df_metrics)
        
        print(f"\n=== Engine Leaderboard (Buffer: {args.buffer_ms}ms) ===")
        # Group by engine and calculate mean scores across all tested files
        summary = (
            df_metrics
            .group_by("engine")
            .agg([
                pl.col("raw_pitch_acc").mean().round(4).alias("mean_rpa"),
                pl.col("voicing_recall").mean().round(4).alias("mean_voicing_recall"),
                pl.col("false_alarm").mean().round(4).alias("mean_false_alarm")
            ])
            .sort("mean_rpa", descending=True)
        )
        print(summary)
    else:
        print("No valid Audio/MIDI pairs evaluated.")


if __name__ == "__main__":
    main()