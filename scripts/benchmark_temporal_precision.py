"""Automated benchmarking suite for temporal precision and acoustic robustness."""

import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from melodict.extraction.sota_models import EngineFactory


def midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def apply_stage_acoustics(audio: np.ndarray, sr: int) -> np.ndarray:
    noise = np.random.normal(0, 0.015, len(audio))
    t = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
    rumble = 0.03 * np.sin(2 * np.pi * 50 * t)
    
    reverb = np.zeros_like(audio)
    delays = [(0.05, 0.5), (0.12, 0.3), (0.25, 0.15)]
    for delay_sec, decay in delays:
        delay_samples = int(delay_sec * sr)
        if delay_samples < len(audio):
            reverb[delay_samples:] += audio[:-delay_samples] * decay
            
    final_audio = audio + noise + rumble + reverb
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio /= max_val
        
    return final_audio.astype(np.float32)


def generate_chromatic_scale(start_midi: int = 45, end_midi: int = 81, duration_sec: float = 0.5, sr: int = 44100, apply_acoustics: bool = True) -> tuple[np.ndarray, list[int]]:
    """Generates a chromatic scale with 5ms ADSR envelopes to measure temporal cutoffs."""
    notes_count = (end_midi - start_midi) + 1
    samples_per_note = int(duration_sec * sr)
    total_samples = notes_count * samples_per_note
    
    audio = np.zeros(total_samples, dtype=np.float32)
    ground_truth = []
    
    for i in range(notes_count):
        current_midi = start_midi + i
        freq = midi_to_freq(current_midi)
        start_idx = i * samples_per_note
        end_idx = start_idx + samples_per_note
        t = np.linspace(0, duration_sec, samples_per_note, endpoint=False)
        
        tone = (
            0.6 * np.sin(2 * np.pi * freq * t) + 
            0.3 * np.sin(2 * np.pi * (freq * 2) * t) + 
            0.1 * np.sin(2 * np.pi * (freq * 3) * t)
        )
        
        fade_samples = int(0.005 * sr)
        envelope = np.ones(samples_per_note)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        audio[start_idx:end_idx] = tone * envelope
        
        # CORRECTED GROUND TRUTH: Assign 0 (silence) during the 5ms ADSR fade windows
        gt_note = np.full(samples_per_note, current_midi)
        gt_note[:fade_samples] = 0
        gt_note[-fade_samples:] = 0
        ground_truth.extend(gt_note.tolist())
        
    final_audio = apply_stage_acoustics(audio, sr) if apply_acoustics else audio
    return final_audio, ground_truth


def run_temporal_benchmark(note_duration_sec: float, buffer_size_ms: int = 46, apply_acoustics: bool = True) -> None:
    sr = 44100
    hop_length = int(sr * (buffer_size_ms / 1000.0))
    start_midi = 45
    end_midi = 81
    
    acoustic_mode = "Stage Acoustics" if apply_acoustics else "Clean Audio"
    print(f"\nGenerating {start_midi} to {end_midi} with {acoustic_mode}...")
    
    audio, ground_truth = generate_chromatic_scale(start_midi, end_midi, note_duration_sec, sr, apply_acoustics)
    
    engines = EngineFactory.get_all_available()
    if not engines:
        print("No engines could be loaded.")
        return

    print(f"\n--- Temporal Precision Benchmark ({note_duration_sec}s/note) ---\n")
    
    results = []
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    
    for engine in engines:
        predicted_track = []
        ground_truth_track = []
        
        print(f"Testing Engine: {engine.name}")
        for i in tqdm(range(0, len(audio), hop_length), desc="Processing Frames"):
            buffer = audio[i : i + hop_length]
            if len(buffer) < hop_length:
                break
                
            pred = engine.predict_frame(buffer, sr)
            pred_pitch = int(pred) if pred is not None else 0
            
            gt_buffer = ground_truth[i : i + hop_length]
            if gt_buffer.count(0) > (len(gt_buffer) * 0.10):
                gt_pitch = 0
            else:
                gt_pitch = max(set(gt_buffer), key=gt_buffer.count)
            
            predicted_track.append(pred_pitch)
            ground_truth_track.append(gt_pitch)

        correct_frames = sum(1 for p, g in zip(predicted_track, ground_truth_track) if p == g)
        accuracy = (correct_frames / len(ground_truth_track)) * 100
        
        delays = []
        for i in range(1, len(ground_truth_track)):
            if ground_truth_track[i] != ground_truth_track[i-1]: 
                frames_delayed = 0
                for j in range(i, len(predicted_track)):
                    if predicted_track[j] == ground_truth_track[i]:
                        break
                    frames_delayed += 1
                delays.append(frames_delayed)
                
        avg_delay_frames = sum(delays) / len(delays) if delays else 0
        avg_delay_ms = avg_delay_frames * buffer_size_ms
        
        results.append({
            "engine": engine.name,
            "accuracy": accuracy,
            "latency_ms": avg_delay_ms,
            "track_pred": predicted_track,
            "track_gt": ground_truth_track
        })
        
        print(f"  - Accuracy: {accuracy:.2f}% | Latency: ~{avg_delay_ms:.1f} ms\n")

    # 1. Export CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_suffix = "acoustic" if apply_acoustics else "clean"
    csv_path = os.path.join(export_dir, f"benchmark_{file_suffix}_{timestamp}.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Engine", "Accuracy (%)", "Transition Latency (ms)"])
        for r in results:
            writer.writerow([r["engine"], f"{r['accuracy']:.2f}", f"{r['latency_ms']:.1f}"])
            
    # 2. Export Plot
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(results[0]["track_gt"])) * (buffer_size_ms / 1000.0)
    plt.plot(time_axis, results[0]["track_gt"], label="Ground Truth", color="black", linestyle="--", linewidth=2)
    
    colors = ['#E94560', '#6C4AB6', '#219F94']
    for idx, r in enumerate(results):
        c = colors[idx % len(colors)]
        plt.plot(time_axis, r["track_pred"], label=f"{r['engine']} (Acc: {r['accuracy']:.1f}%)", color=c, alpha=0.7)
        
    plt.xlabel("Time (seconds)", fontweight='bold')
    plt.ylabel("MIDI Pitch", fontweight='bold')
    plt.title(f"Engine Pitch Tracking Precision ({note_duration_sec}s per note | {acoustic_mode})", fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(export_dir, f"benchmark_{file_suffix}_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved CSV Report : {csv_path}")
    print(f"Saved PNG Plot   : {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark temporal precision and acoustic robustness.")
    parser.add_argument("--note_duration", type=float, default=0.5, help="Seconds per note in the scale.")
    parser.add_argument("--clean", action="store_true", help="Run with clean audio (disable stage acoustics).")
    args = parser.parse_args()
    
    run_temporal_benchmark(args.note_duration, apply_acoustics=not args.clean)