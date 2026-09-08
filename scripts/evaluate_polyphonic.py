"""
Polyphonic Streaming Evaluation Script (Batch Mode, Parallel, Resumable)
Compares the Causal Basic-Pitch engine against an offline Ground Truth MIDI.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import mir_eval
import numpy as np
import pretty_midi
from tqdm import tqdm

# Ensure src/ is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.melodict.extraction.sota_models import EngineFactory


def get_ground_truth_frames(midi_path: str, hop_time: float, max_time: float):
    """Parses a MIDI file and samples active pitches at specific time steps."""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    times = np.arange(0, max_time, hop_time)
    freqs = []
    
    for t in times:
        active_pitches = set()
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                if note.start <= t <= note.end:
                    active_pitches.add(note.pitch)
        
        hz_array = librosa.midi_to_hz(np.array(list(active_pitches))) if active_pitches else np.array([])
        freqs.append(hz_array)
        
    return times, freqs


def process_single_file_task(audio_path: str, midi_path: str, buffer_ms: int):
    """Worker function: Instantiates its own engine and processes one file."""
    # Engine is instantiated per-process because ONNX sessions cannot be pickled safely
    engine = EngineFactory.create("basic-pitch")
    if not engine:
        raise RuntimeError("Failed to load basic-pitch inside worker.")

    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    duration = len(y) / sr
    hop_samples = int(sr * (buffer_ms / 1000.0))
    hop_time = buffer_ms / 1000.0

    est_times = []
    est_freqs = []

    # Create the chunks range
    chunks = range(0, len(y), hop_samples)
    total_chunks = len(chunks)
    
    # Process chunks with a heartbeat print to avoid console clutter
    for idx, i in enumerate(chunks):
        chunk = y[i:i + hop_samples]
        if len(chunk) < hop_samples:
            break
            
        current_time = (i + hop_samples) / sr
        est_times.append(current_time)
        
        active_midi_notes = engine.predict_polyphonic_frame(chunk, sample_rate=sr)
        
        if active_midi_notes:
            hz_array = librosa.midi_to_hz(np.array(active_midi_notes))
        else:
            hz_array = np.array([])
            
        est_freqs.append(hz_array)
        
        # Print a heartbeat every 500 frames so you know it is working
        if idx % 500 == 0 and idx > 0:
            print(f"[{os.path.basename(audio_path)[:20]}...] Processed {idx}/{total_chunks} frames.")

    ref_times, ref_freqs = get_ground_truth_frames(midi_path, hop_time, duration)

    min_len = min(len(ref_times), len(est_times))
    ref_times = ref_times[:min_len]
    ref_freqs = ref_freqs[:min_len]
    est_times = ref_times
    est_freqs = est_freqs[:min_len]

    scores = mir_eval.multipitch.evaluate(
        np.array(ref_times), ref_freqs, 
        np.array(est_times), est_freqs
    )
    return audio_path, scores


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate streaming polyphonic extraction.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing WAV/MIDI pairs")
    parser.add_argument("--exact_file", type=str, default=None, help="Path to a specific .wav file to process")
    parser.add_argument("--buffer_ms", type=int, default=46, help="Streaming buffer size in ms")
    parser.add_argument("--max_files", type=int, default=None, help="Limit the number of files to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--resume", type=str, default="exports/evaluation_results.json", help="JSON file to save/resume progress")
    args = parser.parse_args()

    if not args.data_dir and not args.exact_file:
        print("Error: You must provide either --data_dir or --exact_file")
        sys.exit(1)

    # 1. Determine files to process
    audio_files = []
    if args.exact_file:
        audio_files = [args.exact_file]
    else:
        audio_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
        
    if not audio_files:
        print("No .wav files found.")
        sys.exit(1)

    if args.max_files:
        audio_files = audio_files[:args.max_files]

    # 2. Load resume state
    results = {}
    if os.path.exists(args.resume):
        try:
            with open(args.resume, "r") as f:
                results = json.load(f)
            relevant_results = {k: v for k, v in results.items() if k in audio_files}
            print(f"Resumed session: Found {len(relevant_results)} completed files out of the {len(audio_files)} requested.")
        except json.JSONDecodeError:
            print("Warning: Resume file is corrupted. Starting fresh.")

    # 3. Filter files
    pending_files = [f for f in audio_files if f not in results]

    if not pending_files:
        print("All requested files have already been processed. Skipping to results...")
    else:
        print(f"Processing {len(pending_files)} files using {args.workers} workers...")
    
    if args.max_files:
        pending_files = pending_files[:args.max_files]

    if not pending_files:
        print("All requested files have already been processed. Skipping to results...")
    else:
        print(f"Processing {len(pending_files)} files using {args.workers} workers...")

        # 4. Parallel Processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit tasks
            futures = {}
            for audio_path in pending_files:
                base_name = os.path.splitext(audio_path)[0]
                midi_path = f"{base_name}.midi" if os.path.exists(f"{base_name}.midi") else f"{base_name}.mid"
                
                if not os.path.exists(midi_path):
                    print(f"[Warning] No MIDI found for {os.path.basename(audio_path)}. Skipping.")
                    continue
                    
                future = executor.submit(process_single_file_task, audio_path, midi_path, args.buffer_ms)
                futures[future] = audio_path

            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Batch Progress"):
                audio_path = futures[future]
                try:
                    processed_path, scores = future.result()
                    results[processed_path] = scores
                    
                    # Auto-save after every file finishes
                    with open(args.resume, "w") as f:
                        json.dump(results, f, indent=4)
                        
                except Exception as e:  # noqa: BLE001
                    print(f"\n[Error] Failed on {os.path.basename(audio_path)}: {e}")

    # 5. Calculate final metrics from the loaded JSON
    if results:
        avg_precision = np.mean([s['Precision'] for s in results.values()])
        avg_recall = np.mean([s['Recall'] for s in results.values()])
        avg_accuracy = np.mean([s['Accuracy'] for s in results.values()])
        
        # Calculate F1-Score manually since mir_eval multipitch omits it
        f1_scores = []
        for s in results.values():
            p = s['Precision']
            r = s['Recall']
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            f1_scores.append(f1)
            
        avg_f1 = np.mean(f1_scores)

        print("\n" + "="*60)
        print("         BATCH POLYPHONIC EVALUATION RESULTS (STREAMING)         ")
        print("="*60)
        print(f"Total files evaluated: {len(results)}")
        print(f"Average F-Measure (F1): {avg_f1:.4f}")
        print(f"Average Precision:      {avg_precision:.4f}")
        print(f"Average Recall:         {avg_recall:.4f}")
        print(f"Average Accuracy:       {avg_accuracy:.4f}")
        print("="*60)

if __name__ == "__main__":
    main()