"""
Export Basic-Pitch Offline
Processes WAV files through basic-pitch in full acausal (offline) mode and saves them as MIDI.
"""
import argparse
import glob
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

from tqdm import tqdm

# Set environment variables for ONNX before importing
os.environ["BASIC_PITCH_TF"] = "0"
os.environ["BASIC_PITCH_ONNX"] = "1"
os.environ["BASIC_PITCH_COREML"] = "0"

import logging

logging.getLogger().setLevel(logging.ERROR)

from basic_pitch.inference import predict


def main():
    parser = argparse.ArgumentParser(description="Export entire WAVs to MIDI using Basic-Pitch (Offline).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing WAV files")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the MIDI files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    
    if not audio_files:
        print(f"No .wav files found in {args.data_dir}")
        sys.exit(1)

    print(f"Starting Offline extraction for {len(audio_files)} files using ONNX backend...")

    for audio_path in tqdm(audio_files, desc="Exporting MIDIs"):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_midi_path = os.path.join(args.out_dir, f"{base_name}_basicpitch.mid")
        
        if os.path.exists(out_midi_path):
            continue
            
        try:
            # Predict on the entire file at once (Acausal mode) and silence the prints
            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
                _, midi_data, _ = predict(audio_path)
                
            midi_data.write(out_midi_path)
        except Exception as e:
            print(f"\n[Error] Failed to process {base_name}: {e}")

    print("\nExtraction complete! MIDIs saved to:", args.out_dir)


if __name__ == "__main__":
    main()