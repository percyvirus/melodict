"""Advanced DSP benchmarking suite for pitch tracking engines with individual subplots."""

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
    """Injects live-environment noise, 50Hz rumble, and room reverb."""
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


def analyze_dsp_metrics(pred: list[int], gt: list[int], buffer_ms: int) -> dict:
    """Calculates advanced pitch tracking metrics."""
    total_frames = len(gt)
    correct = 0
    octave_errors = 0
    voicing_false_alarms = 0
    missed_voicings = 0
    delays_frames = []
    
    for i in range(total_frames):
        p, g = pred[i], gt[i]
        
        if p == g:
            correct += 1
        elif p > 0 and g > 0 and abs(p - g) % 12 == 0:
            octave_errors += 1
        elif p > 0 and g == 0:
            voicing_false_alarms += 1
        elif p == 0 and g > 0:
            missed_voicings += 1

    # Calculate Transition Latency
    for i in range(1, total_frames):
        if gt[i] != gt[i-1] and gt[i] > 0:  
            lag = 0
            for j in range(i, min(i + 20, total_frames)): 
                if pred[j] == gt[i]:
                    break
                lag += 1
            delays_frames.append(lag)

    avg_lag_ms = (sum(delays_frames) / len(delays_frames)) * buffer_ms if delays_frames else 0.0

    return {
        "accuracy": (correct / total_frames) * 100,
        "octave_error_rate": (octave_errors / total_frames) * 100,
        "false_alarm_rate": (voicing_false_alarms / total_frames) * 100,
        "missed_voicing_rate": (missed_voicings / total_frames) * 100,
        "transition_lag_ms": avg_lag_ms
    }


def run_dsp_benchmark(note_duration_sec: float, buffer_size_ms: int = 46, apply_acoustics: bool = True) -> None:
    sr = 44100
    hop_length = int(sr * (buffer_size_ms / 1000.0))
    start_midi, end_midi = 45, 81
    
    acoustic_mode = "Stage Acoustics" if apply_acoustics else "Clean Audio"
    print(f"\nGenerating {start_midi} to {end_midi} (DSP Analysis Mode) with {acoustic_mode}...")
    
    audio, ground_truth = generate_chromatic_scale(start_midi, end_midi, note_duration_sec, sr, apply_acoustics)
    
    engines = EngineFactory.get_all_available()
    if not engines:
        print("No engines could be loaded.")
        return

    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    
    results = []
    
    for engine in engines:
        predicted_track = []
        ground_truth_track = []
        
        print(f"Testing Engine: {engine.name}")
        for i in tqdm(range(0, len(audio), hop_length), desc=f"Processing {engine.name}"):
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

        metrics = analyze_dsp_metrics(predicted_track, ground_truth_track, buffer_size_ms)
        metrics["engine"] = engine.name
        metrics["track_pred"] = predicted_track
        metrics["track_gt"] = ground_truth_track
        results.append(metrics)
        
        print(f"  Accuracy: {metrics['accuracy']:.1f}% | Octave Err: {metrics['octave_error_rate']:.1f}% | Lag: {metrics['transition_lag_ms']:.1f}ms\n")

    # 1. Export CSV Metrics
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_suffix = "acoustic" if apply_acoustics else "clean"
    csv_path = os.path.join(export_dir, f"dsp_benchmark_{file_suffix}_{timestamp}.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Engine", "Accuracy (%)", "Octave Errors (%)", "Missed Voicing (%)", "False Alarms (%)", "Lag (ms)"])
        for r in results:
            writer.writerow([
                r["engine"], f"{r['accuracy']:.2f}", f"{r['octave_error_rate']:.2f}", 
                f"{r['missed_voicing_rate']:.2f}", f"{r['false_alarm_rate']:.2f}", f"{r['transition_lag_ms']:.1f}"
            ])

    # 2. Export Subplot Visualization
    num_engines = len(results)
    fig, axes = plt.subplots(nrows=num_engines, ncols=1, figsize=(14, 4 * num_engines), sharex=True)
    if num_engines == 1:
        axes = [axes]
        
    time_axis = np.arange(len(results[0]["track_gt"])) * (buffer_size_ms / 1000.0)
    
    colors = ['#E94560', '#6C4AB6', '#219F94', '#F39C12']
    
    for idx, r in enumerate(results):
        ax = axes[idx]
        c = colors[idx % len(colors)]
        
        ax.plot(time_axis, r["track_gt"], label="Ground Truth", color="black", linestyle="--", linewidth=2, alpha=0.8)
        ax.plot(time_axis, r["track_pred"], label=f"{r['engine']} Prediction", color=c, linewidth=1.5, alpha=0.9)
        
        # Highlight Octave Errors visually
        pred_arr = np.array(r["track_pred"])
        gt_arr = np.array(r["track_gt"])
        octave_err_mask = (pred_arr > 0) & (gt_arr > 0) & (np.abs(pred_arr - gt_arr) % 12 == 0) & (pred_arr != gt_arr)
        
        if np.any(octave_err_mask):
            ax.scatter(time_axis[octave_err_mask], pred_arr[octave_err_mask], color='red', marker='x', label='Octave Error', zorder=5)

        title_str = f"{r['engine']} - Acc: {r['accuracy']:.1f}% | Octave Err: {r['octave_error_rate']:.1f}% | Lag: {r['transition_lag_ms']:.1f}ms"
        ax.set_title(title_str, fontweight='bold')
        ax.set_ylabel("MIDI Pitch")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time (seconds)", fontweight='bold')
    plt.suptitle(f"Engine Pitch Tracking Precision ({note_duration_sec}s per note | {acoustic_mode})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(export_dir, f"dsp_benchmark_{file_suffix}_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved DSP CSV Report : {csv_path}")
    print(f"Saved Subplot PNG    : {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced DSP benchmark for pitch tracking.")
    parser.add_argument("--note_duration", type=float, default=5.0, help="Seconds per note in the scale.")
    parser.add_argument("--clean", action="store_true", help="Run with clean audio (disable stage acoustics).")
    args = parser.parse_args()
    
    run_dsp_benchmark(args.note_duration, apply_acoustics=not args.clean)