"""
Audio-to-MIDI Melody Evaluation Pipeline for GuitarSet.
Parses .jams annotations, matches them with corresponding audio, 
and benchmarks SOTA engines using Smart Skyline extraction.
"""

import argparse
from pathlib import Path

import jams
import librosa
import mir_eval
import numpy as np
import polars as pl
from scipy.ndimage import maximum_filter1d
from scipy.signal import medfilt
from tqdm import tqdm

from melodict.extraction.sota_models import EngineFactory, PitchExtractorEngine


def load_ground_truth_jams_skyline(
    jams_path: Path, 
    max_duration: float = 15.0, 
    drop_threshold_semitones: float = 12.0, 
    window_sec: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Parses GuitarSet JAMS annotations into a Smart Skyline melody trajectory."""
    jam = jams.load(str(jams_path))
    
    dt = 0.01
    time_grid = np.arange(0, max_duration, dt)
    skyline_pitches = np.zeros_like(time_grid)
    
    # GuitarSet stores notes in 'note_midi' namespaces (one per string usually)
    note_annos = jam.search(namespace='note_midi')
    
    for anno in note_annos:
        for obs in anno.data:
            start_time = obs.time
            end_time = obs.time + obs.duration
            pitch = obs.value
            
            if start_time >= max_duration:
                continue
            end_time = min(end_time, max_duration)
            
            start_idx = int(start_time / dt)
            end_idx = int(end_time / dt)
            
            if start_idx < len(skyline_pitches):
                skyline_pitches[start_idx:end_idx] = np.maximum(
                    skyline_pitches[start_idx:end_idx], pitch
                )
                
    # Smart Skyline Filtering (Melodic Ceiling)
    window_size_frames = int(window_sec / dt)
    melodic_ceiling = maximum_filter1d(skyline_pitches, size=window_size_frames, mode='constant', cval=0.0)
    
    smart_skyline_pitches = np.copy(skyline_pitches)
    drop_mask = (skyline_pitches > 0) & (skyline_pitches < (melodic_ceiling - drop_threshold_semitones))
    smart_skyline_pitches[drop_mask] = 0.0
    
    smart_freqs = np.zeros_like(smart_skyline_pitches)
    smart_active = smart_skyline_pitches > 0
    smart_freqs[smart_active] = librosa.midi_to_hz(smart_skyline_pitches[smart_active])
    
    return time_grid, smart_freqs


def predict_realtime(engine: PitchExtractorEngine, audio_path: Path, max_duration: float, buffer_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulates real-time sliding window extraction."""
    sample_rate = 44100
    hop_length = int(sample_rate * (buffer_ms / 1000.0))
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=max_duration)
    
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


def evaluate_guitarset(dataset_dir: Path, audio_type: str, max_files: int = 5, max_duration: float = 10.0, buffer_ms: int = 46, context_sec: float = 2.0) -> pl.DataFrame:
    engines = EngineFactory.get_all_available()
    if not engines:
        print("No pitch extraction engines available.")
        return pl.DataFrame()
        
    for engine in engines:
        if engine.context_sec > 0:
            engine.context_sec = context_sec
            
    print(f"\nLoaded {len(engines)} engines for GuitarSet Benchmarking.")

    results = []
    
    # Dynamically select the audio suffix based on user input
    suffix = f"_{audio_type}.wav"
    audio_files = list(dataset_dir.rglob(f"*{suffix}"))[:max_files]
    
    if not audio_files:
        print(f"No '{suffix}' files found in {dataset_dir}.")
        return pl.DataFrame()

    print(f"Evaluating {len(audio_files)} GuitarSet files (Audio: {audio_type} | Buffer: {buffer_ms}ms | Context: {context_sec}s)...\n")
    
    for audio_path in tqdm(audio_files, desc="Processing Guitar Tracks"):
        # Strip the specific suffix to find the matching JAMS file
        base_name = audio_path.name.replace(suffix, ".jams")
        jams_path = next(dataset_dir.rglob(base_name), None)
        
        if not jams_path:
            print(f"Warning: Missing annotation for {audio_path.name}")
            continue
            
        ref_times, ref_freqs = load_ground_truth_jams_skyline(jams_path, max_duration)
        if len(ref_times) == 0:
            continue
            
        for engine in engines:
            est_times, est_freqs = predict_realtime(engine, audio_path, max_duration, buffer_ms)
            
            if len(est_times) > 0:
                metrics = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
                rpa = metrics.get('Raw Pitch Accuracy', 0.0)
                vr = metrics.get('Voicing Recall', 0.0)
                vfa = metrics.get('Voicing False Alarm', 0.0)
            else:
                rpa, vr, vfa = 0.0, 0.0, 0.0
                
            results.append({
                "engine": engine.name,
                "file_name": audio_path.name,
                "rpa": round(float(rpa), 4),
                "vr": round(float(vr), 4),
                "vfa": round(float(vfa), 4)
            })
            
    return pl.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Melodict Engines on GuitarSet.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to GuitarSet dataset folder.")
    parser.add_argument("--max_files", type=int, default=10, help="Number of files to process.")
    parser.add_argument("--duration", type=float, default=15.0, help="Seconds per file (default 15.0).")
    parser.add_argument(
        "--audio_type", 
        type=str, 
        choices=["mic", "mix", "hex", "hex_cln"], 
        default="mic", 
        help="Type of guitar audio to evaluate (mic=air, mix=line-in, hex=hexaphonic)."
    )
    
    args = parser.parse_args()
    data_path = Path(args.data_dir)
    
    if not data_path.exists():
        print(f"Error: Dataset directory not found at {data_path}")
        return
        
    df_metrics = evaluate_guitarset(
        data_path, 
        audio_type=args.audio_type, 
        max_files=args.max_files, 
        max_duration=args.duration, 
        buffer_ms=46, 
        context_sec=2.0
    )
    
    if len(df_metrics) > 0:
        print("\n" + "="*70)
        print(f"🎸 GUITARSET LEADERBOARD (Audio: {args.audio_type.upper()} | Buffer: 46ms | Context: 2.0s) 🎸")
        print("="*70)
        
        summary = (
            df_metrics
            .group_by("engine")
            .agg([
                pl.col("rpa").mean().round(4).alias("mean_RPA"),
                pl.col("vr").mean().round(4).alias("mean_Recall"),
                pl.col("vfa").mean().round(4).alias("mean_FalseAlarm")
            ])
            .sort("mean_RPA", descending=True)
        )
        print(summary)
    else:
        print("No valid Audio/JAMS pairs evaluated.")


if __name__ == "__main__":
    main()