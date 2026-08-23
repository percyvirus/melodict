"""
Audio-to-MIDI Melody Visualization Tool for Melodict.
Includes Smart Skyline Extraction for perfect Ground Truth generation.
"""

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pretty_midi
from scipy.ndimage import maximum_filter1d
from scipy.signal import medfilt

from melodict.extraction.sota_models import EngineFactory, PitchExtractorEngine


def predict_with_realtime_engine_melody(
    engine: PitchExtractorEngine, 
    audio_path: Path, 
    max_duration: float = 15.0, 
    buffer_ms: int = 46
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates real-time processing and returns smoothed time-frequency trajectories."""
    sample_rate = 44100
    hop_length = int(sample_rate * (buffer_ms / 1000.0))
    
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=max_duration)
    
    est_times = []
    est_freqs = []
    
    engine.reset_context()
    
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
            est_freqs.append(0.0)
            
    est_times = np.array(est_times)
    est_freqs = np.array(est_freqs)
    
    if len(est_freqs) >= 5:
        est_freqs = medfilt(est_freqs, kernel_size=5)
        
    return est_times, est_freqs


def load_ground_truth_smart_skyline(
    midi_path: Path, 
    max_duration: float = 15.0, 
    drop_threshold_semitones: float = 12.0, 
    window_sec: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a smart time-frequency trajectory, filtering out bass bleeding 
    during melodic rests using offline contextual smoothing.
    Returns: time_grid, raw_skyline_freqs, smart_skyline_freqs
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    
    dt = 0.01  # 10ms high-resolution grid
    time_grid = np.arange(0, max_duration, dt)
    skyline_pitches = np.zeros_like(time_grid)
    
    # 1. Standard Skyline: Highest pitch at any given time
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
                    
    # 2. Smart Skyline: Offline Contextual Filtering
    window_size_frames = int(window_sec / dt)
    
    # Create a rolling maximum envelope (Melodic Ceiling)
    melodic_ceiling = maximum_filter1d(skyline_pitches, size=window_size_frames, mode='constant', cval=0.0)
    
    # Identify moments where the pitch drops drastically below the ceiling (harmony bleeding)
    smart_skyline_pitches = np.copy(skyline_pitches)
    drop_mask = (skyline_pitches > 0) & (skyline_pitches < (melodic_ceiling - drop_threshold_semitones))
    
    # Erase those harmony notes, turning them into proper melodic rests
    smart_skyline_pitches[drop_mask] = 0.0
    
    # Convert to Hz arrays
    raw_freqs = np.zeros_like(skyline_pitches)
    raw_active = skyline_pitches > 0
    raw_freqs[raw_active] = librosa.midi_to_hz(skyline_pitches[raw_active])
    
    smart_freqs = np.zeros_like(smart_skyline_pitches)
    smart_active = smart_skyline_pitches > 0
    smart_freqs[smart_active] = librosa.midi_to_hz(smart_skyline_pitches[smart_active])
    
    return time_grid, raw_freqs, smart_freqs


def freq_to_midi_safe(freqs: np.ndarray) -> np.ndarray:
    """Converts Hz to MIDI notes, turning 0.0 into NaNs for clean matplotlib plotting."""
    midi_notes = np.zeros_like(freqs)
    active = freqs > 0
    midi_notes[active] = librosa.hz_to_midi(freqs[active])
    midi_notes[~active] = np.nan
    return midi_notes


def plot_melody_comparison(
    audio_path: Path, 
    max_duration: float = 10.0, 
    buffer_ms: int = 46, 
    context_sec: float = 1.0, 
    output_path: str = None
):
    midi_path = audio_path.with_suffix(".midi")
    if not midi_path.exists():
        midi_path = audio_path.with_suffix(".mid")
    
    if not midi_path.exists():
        print(f"Error: Could not find MIDI file for {audio_path}")
        return

    engines = EngineFactory.get_all_available()
    if not engines:
        print("Error: No pitch extraction engines available.")
        return
        
    for engine in engines:
        if engine.context_sec > 0:
            engine.context_sec = context_sec

    print(f"Processing {audio_path.name}...")
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    
    # Get both the raw and the smart skyline for visualization
    ref_times, raw_ref_freqs, smart_ref_freqs = load_ground_truth_smart_skyline(midi_path, max_duration)
    
    raw_ref_midi_notes = freq_to_midi_safe(raw_ref_freqs)
    smart_ref_midi_notes = freq_to_midi_safe(smart_ref_freqs)
    
    num_plots = len(engines)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(24, 4 * num_plots), sharex=True, sharey=True)
    if num_plots == 1:
        axes = [axes]
        
    colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
    
    for idx, (engine, ax, color) in enumerate(zip(engines, axes, colors)):
        print(f"Running {engine.name}...")
        
        # Layer 1: Full Ground Truth (Harmony/Chords)
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start < max_duration:
                        end_time = min(note.end, max_duration)
                        ax.plot([note.start, end_time], [note.pitch, note.pitch], 
                                color='lightgray', linewidth=6, solid_capstyle='butt', alpha=0.5)
        
        # Layer 2: Raw Skyline (Shows the errors dropping to bass)
        ax.plot(ref_times, raw_ref_midi_notes, color='red', linewidth=2, linestyle=':', label='Raw Skyline (Bass Bleeding)', alpha=0.6)
        
        # Layer 3: Smart Skyline (The filtered, clean melody)
        ax.plot(ref_times, smart_ref_midi_notes, color='black', linewidth=3, linestyle='-', label='Smart Skyline (True Melody)')
        
        # Layer 4: Model Prediction
        est_times, est_freqs = predict_with_realtime_engine_melody(engine, audio_path, max_duration, buffer_ms)
        est_midi_notes = freq_to_midi_safe(est_freqs)
        
        ax.plot(est_times, est_midi_notes, color=color, linewidth=2.5, label=f'{engine.name} Output')
        
        # Formatting
        ax.set_title(f"Engine: {engine.name} (Buffer: {buffer_ms}ms, Context: {engine.context_sec}s)", fontsize=14, fontweight='bold')
        ax.set_ylabel("MIDI Note", fontsize=12)
        ax.grid(True, linestyle='-', alpha=0.3)
        
        custom_lines = [
            Line2D([0], [0], color='lightgray', lw=6),
            Line2D([0], [0], color='red', lw=2, linestyle=':'),
            Line2D([0], [0], color='black', lw=3, linestyle='-'),
            Line2D([0], [0], color=color, lw=2.5)
        ]
        ax.legend(custom_lines, ['Full MIDI (Harmony)', 'Raw Skyline (Errors)', 'Smart Skyline GT (Melody)', f'{engine.name} Output'], loc='upper right', fontsize=11)

    axes[-1].set_xlabel("Time (seconds)", fontsize=14)
    plt.xlim(0, max_duration)
    
    active_pitches = [n.pitch for i in midi_data.instruments for n in i.notes if n.start < max_duration]
    if active_pitches:
        plt.ylim(min(active_pitches) - 5, max(active_pitches) + 5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved successfully to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Melodict Engines with Smart Skyline GT.")
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--buffer_ms", type=int, default=46)
    parser.add_argument("--context_sec", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="melody_comparison_plot.png")
    
    args = parser.parse_args()
    audio_path = Path(args.audio_file)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return
        
    plot_melody_comparison(
        audio_path, 
        max_duration=args.duration,
        buffer_ms=args.buffer_ms,
        context_sec=args.context_sec,
        output_path=args.output
    )


if __name__ == "__main__":
    main()