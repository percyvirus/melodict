"""
Export Basic-Pitch Streaming (Causal)
Simulates the 46ms causal streaming pipeline and reconstructs the output into MIDI files.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import pretty_midi
from tqdm import tqdm

# Ensure src/ is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.melodict.extraction.sota_models import EngineFactory


def process_single_streaming_export(audio_path: str, out_midi_path: str, buffer_ms: int):
    """Worker function to simulate streaming and build a MIDI file frame by frame."""
    engine = EngineFactory.create("basic-pitch")
    if not engine:
        raise RuntimeError("Failed to load basic-pitch inside worker.")

    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    hop_samples = int(sr * (buffer_ms / 1000.0))
    
    # State tracker for MIDI note generation
    active_notes_state = {}  # Format: {midi_pitch: start_time}
    midi_events = []         # Format: (pitch, start_time, end_time)

    chunks = range(0, len(y), hop_samples)
    for i in chunks:
        chunk = y[i:i + hop_samples]
        if len(chunk) < hop_samples:
            break
            
        current_time = i / sr
        chunk_end_time = (i + hop_samples) / sr
        
        # Get active pitches for this exact 46ms window
        active_pitches = engine.predict_polyphonic_frame(chunk, sample_rate=sr)
        
        # 1. Check for newly played notes (Note ON)
        for p in active_pitches:
            if p not in active_notes_state:
                active_notes_state[p] = current_time
                
        # 2. Check for ended notes (Note OFF)
        ended_pitches = []
        for p in active_notes_state:
            if p not in active_pitches:
                ended_pitches.append(p)
                
        for p in ended_pitches:
            start_t = active_notes_state.pop(p)
            # Avoid zero-duration notes
            if chunk_end_time > start_t:
                midi_events.append((p, start_t, chunk_end_time))

    # Flush any remaining notes that were active when the audio ended
    final_time = len(y) / sr
    for p, start_t in active_notes_state.items():
        if final_time > start_t:
            midi_events.append((p, start_t, final_time))

    # Construct the actual MIDI file
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # Program 0 = Acoustic Grand Piano
    
    for pitch, start, end in midi_events:
        # Default velocity to 100 as we only track pitch presence, not dynamics
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        inst.notes.append(note)
        
    pm.instruments.append(inst)
    pm.write(out_midi_path)
    
    return audio_path


def main():
    parser = argparse.ArgumentParser(description="Export streaming WAVs to MIDI using Basic-Pitch (Causal).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing WAV files")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the MIDI files")
    parser.add_argument("--buffer_ms", type=int, default=46, help="Streaming buffer size in ms")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    
    if not audio_files:
        print(f"No .wav files found in {args.data_dir}")
        sys.exit(1)

    # Filter files that are already exported
    pending_files = []
    for audio_path in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_midi_path = os.path.join(args.out_dir, f"{base_name}_streaming.mid")
        if not os.path.exists(out_midi_path):
            pending_files.append((audio_path, out_midi_path))

    if not pending_files:
        print(f"All {len(audio_files)} files have already been exported to {args.out_dir}.")
        sys.exit(0)

    print(f"Starting Streaming export for {len(pending_files)} files using {args.workers} workers...")
    print("Note: This will take a while due to real-time I/O simulation overhead.")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_streaming_export, audio_path, out_midi, args.buffer_ms): audio_path 
            for audio_path, out_midi in pending_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Exporting MIDIs"):
            try:
                future.result()
            except Exception as e:  # noqa: BLE001
                audio_path = futures[future]
                print(f"\n[Error] Failed to process {os.path.basename(audio_path)}: {e}")

    print("\nStreaming extraction complete! MIDIs saved to:", args.out_dir)


if __name__ == "__main__":
    main()