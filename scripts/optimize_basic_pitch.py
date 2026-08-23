"""
Hyperparameter Optimization and Offline Upper Bound for Basic-Pitch.
Grid searches the best combination of buffer_ms and context_sec for real-time tracking.
Includes togglable offline extraction to find the theoretical upper bound.
"""

import argparse
import itertools
import os
import tempfile
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import polars as pl
import pretty_midi
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import medfilt
from tqdm import tqdm

from melodict.extraction.sota_models import EngineFactory, PitchExtractorEngine


def load_ground_truth_smart_skyline(
    midi_path: Path, 
    max_duration: float = 15.0, 
    drop_threshold_semitones: float = 12.0, 
    window_sec: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generates the Smart Skyline Ground Truth (filters out bass bleeding)."""
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    
    dt = 0.01
    time_grid = np.arange(0, max_duration, dt)
    skyline_pitches = np.zeros_like(time_grid)
    
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                if note.start >= max_duration:
                    continue
                end_time = min(note.end, max_duration)
                
                start_idx = int(note.start / dt)
                end_idx = int(end_time / dt)
                
                if start_idx < len(skyline_pitches):
                    skyline_pitches[start_idx:end_idx] = np.maximum(
                        skyline_pitches[start_idx:end_idx], note.pitch
                    )
                    
    window_size_frames = int(window_sec / dt)
    melodic_ceiling = maximum_filter1d(skyline_pitches, size=window_size_frames, mode='constant', cval=0.0)
    
    smart_skyline_pitches = np.copy(skyline_pitches)
    drop_mask = (skyline_pitches > 0) & (skyline_pitches < (melodic_ceiling - drop_threshold_semitones))
    smart_skyline_pitches[drop_mask] = 0.0
    
    smart_freqs = np.zeros_like(smart_skyline_pitches)
    smart_active = smart_skyline_pitches > 0
    smart_freqs[smart_active] = librosa.midi_to_hz(smart_skyline_pitches[smart_active])
    
    return time_grid, smart_freqs


def predict_realtime(engine: PitchExtractorEngine, audio_path: Path, max_duration: float, buffer_ms: int, context_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulates real-time sliding window extraction."""
    sample_rate = 44100
    hop_length = int(sample_rate * (buffer_ms / 1000.0))
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=max_duration)
    
    engine.context_sec = context_sec
    engine.reset_context()
    
    est_times = []
    est_freqs = []
    
    for i in range(0, len(y), hop_length):
        buffer = y[i : i + hop_length]
        if len(buffer) < hop_length:
            break  
            
        current_time_sec = i / sample_rate
        predicted_midi = engine.predict_frame(buffer, sample_rate)
        
        est_times.append(current_time_sec)
        est_freqs.append(librosa.midi_to_hz(predicted_midi) if predicted_midi is not None else 0.0)
            
    est_freqs = np.array(est_freqs)
    if len(est_freqs) >= 5:
        est_freqs = medfilt(est_freqs, kernel_size=5)
        
    return np.array(est_times), est_freqs


def predict_offline_upper_bound(audio_path: Path, max_duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Processes the entire audio file at once to find Basic-Pitch's maximum theoretical accuracy."""
    from basic_pitch.inference import predict
    from contextlib import redirect_stdout, redirect_stderr
    
    sample_rate = 44100
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=max_duration)
    
    dt = 0.01
    time_grid = np.arange(0, max_duration, dt)
    skyline_pitches = np.zeros_like(time_grid)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
        sf.write(temp_wav.name, y, sample_rate)
        
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                _, midi_data, _ = predict(temp_wav.name)
        
        if midi_data.instruments:
            for note in midi_data.instruments[0].notes:
                if note.start >= max_duration:
                    continue
                end_time = min(note.end, max_duration)
                
                start_idx = int(note.start / dt)
                end_idx = int(end_time / dt)
                
                if start_idx < len(skyline_pitches):
                    skyline_pitches[start_idx:end_idx] = np.maximum(
                        skyline_pitches[start_idx:end_idx], note.pitch
                    )
                    
    est_freqs = np.zeros_like(skyline_pitches)
    active = skyline_pitches > 0
    est_freqs[active] = librosa.midi_to_hz(skyline_pitches[active])
    
    return time_grid, est_freqs


def main():
    parser = argparse.ArgumentParser(description="Grid Search Optimization for Basic-Pitch.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MAESTRO dataset.")
    parser.add_argument("--max_files", type=int, default=10, help="Number of files to process.")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per file.")
    parser.add_argument("--disable_offline", action="store_true", help="Skip the Offline Upper Bound calculation to save time.")
    parser.add_argument("--report_out", type=str, default="basic_pitch_optimization_report.csv", help="Path to save the summary CSV report.")
    
    # NEW ARGUMENTS FOR HYPERPARAMETERS
    parser.add_argument("--buffer_sizes", type=str, default="23,46,92", 
                        help="Comma-separated list of buffer sizes in ms (e.g. 23,46,92).")
    parser.add_argument("--context_sizes", type=str, default="0.5,1.0,1.5,2.0", 
                        help="Comma-separated list of context window sizes in seconds (e.g. 0.5,1.0,2.0).")
    
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Dataset directory not found at {data_path}")
        return

    # Parse the comma-separated strings into lists of numbers
    try:
        buffer_sizes = [int(b.strip()) for b in args.buffer_sizes.split(",")]
        context_sizes = [float(c.strip()) for c in args.context_sizes.split(",")]
    except ValueError as e:
        print(f"Error parsing hyperparameters: {e}. Please ensure they are comma-separated numbers.")
        return
    
    engine = EngineFactory.create("basic-pitch")
    if not engine:
        return
        
    audio_files = list(data_path.rglob("*.wav"))[:args.max_files]
    results = []

    print(f"\nStarting Grid Search on {len(audio_files)} files...")
    
    total_combinations = len(buffer_sizes) * len(context_sizes)
    print(f"Testing {total_combinations} real-time combinations per file.")
    print(f"Buffers (ms): {buffer_sizes}")
    print(f"Contexts (s): {context_sizes}")
    
    if not args.disable_offline:
        print("Offline condition (Ceiling) is ENABLED.")
    else:
        print("Offline condition (Ceiling) is DISABLED.")
    
    for audio_path in tqdm(audio_files, desc="Processing Files"):
        midi_path = audio_path.with_suffix(".midi")
        if not midi_path.exists():
            midi_path = audio_path.with_suffix(".mid")
        if not midi_path.exists():
            continue
            
        ref_times, ref_freqs = load_ground_truth_smart_skyline(midi_path, args.duration)
        if len(ref_times) == 0:
            continue
            
        # 1. Test Offline Upper Bound (Full File) if not disabled
        if not args.disable_offline:
            off_times, off_freqs = predict_offline_upper_bound(audio_path, args.duration)
            metrics_off = mir_eval.melody.evaluate(ref_times, ref_freqs, off_times, off_freqs)
            
            results.append({
                "mode": "Offline (Ceiling)",
                "buffer_ms": "N/A",
                "context_sec": "N/A",
                "rpa": metrics_off.get('Raw Pitch Accuracy', 0.0),
                "vr": metrics_off.get('Voicing Recall', 0.0),
                "vfa": metrics_off.get('Voicing False Alarm', 0.0)
            })

        # 2. Test Real-Time Grid Combinations
        for b_ms, c_sec in itertools.product(buffer_sizes, context_sizes):
            rt_times, rt_freqs = predict_realtime(engine, audio_path, args.duration, b_ms, c_sec)
            
            if len(rt_times) > 0:
                metrics_rt = mir_eval.melody.evaluate(ref_times, ref_freqs, rt_times, rt_freqs)
                results.append({
                    "mode": "Real-Time",
                    "buffer_ms": str(b_ms),
                    "context_sec": str(c_sec),
                    "rpa": metrics_rt.get('Raw Pitch Accuracy', 0.0),
                    "vr": metrics_rt.get('Voicing Recall', 0.0),
                    "vfa": metrics_rt.get('Voicing False Alarm', 0.0)
                })

    if not results:
        print("No valid data processed.")
        return

    df = pl.DataFrame(results)
    
    # Aggregate and sort results to find the best configuration
    summary = (
        df.group_by(["mode", "buffer_ms", "context_sec"])
        .agg([
            pl.col("rpa").mean().round(4).alias("mean_RPA"),
            pl.col("vr").mean().round(4).alias("mean_Recall"),
            pl.col("vfa").mean().round(4).alias("mean_FalseAlarm")
        ])
        .sort("mean_RPA", descending=True)
    )
    
    print("\n" + "="*70)
    print("🏆 BASIC-PITCH HYPERPARAMETER LEADERBOARD (Sorted by Accuracy) 🏆")
    print("="*70)
    print(summary)

    # Save report to CSV
    summary.write_csv(args.report_out)
    print(f"\n✅ Report successfully saved to: {args.report_out}")


if __name__ == "__main__":
    main()