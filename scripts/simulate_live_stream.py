"""Simulate a live symbolic MIDI note stream to test LBDM phrase segmentation and dictionary building."""

import argparse
import time
from typing import List, Tuple
from melodict.segmentation.phrase_builder import PhraseBuilder


def get_test_melody() -> List[Tuple[int, float]]:
    """
    Generate a synthetic melodic stream with distinct musical phrases.
    Returns a list of tuples: (MIDI Note Number, Duration in seconds before next note).
    
    Structure:
    - Phrase 1: Fast ascending jazz line ending with a long rest (time Gestalt boundary).
    - Phrase 2: Mid-register melody with a sudden octave leap (pitch Gestalt boundary).
    - Phrase 3: Short concluding lick.
    """
    return [
        # Phrase 1: Ascending line (C4 to G4) followed by a 1.2-second rest
        (60, 0.2), (62, 0.2), (64, 0.2), (65, 0.2), (67, 1.2),
        
        # Phrase 2: Melody with an abrupt leap of 14 semitones (G4 to A5)
        (67, 0.3), (65, 0.3), (64, 0.3), (78, 0.4),
        
        # Phrase 3: Descending blues resolution
        (76, 0.2), (74, 0.2), (72, 0.2), (70, 0.2), (67, 1.0),
    ]


def run_simulation(speed_multiplier: float = 1.0, threshold: float = 1.5) -> None:
    """Run the live symbolic stream simulation and monitor dictionary generation."""
    print(f"\n--- Melodict Symbolic Phrase Segmentation Test (Speed: {speed_multiplier}x) ---")
    print("Simulating live MIDI stream from instrument...")
    print("Listening for Gestalt boundaries (Rests > 0.8s or Pitch Leaps >= 12 semitones)...\n")

    builder = PhraseBuilder(boundary_threshold=threshold, max_silence_sec=0.8)
    melody = get_test_melody()
    
    start_time = time.time()
    for i, (note, duration) in enumerate(melody, 1):
        elapsed = time.time() - start_time
        freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
        print(f"[{elapsed:05.2f}s] Note Played: MIDI {note} ({freq_hz:.1f} Hz)")

        # Feed note to the live segmenter
        boundary_detected = builder.add_note(note)
        
        if boundary_detected:
            print("          |---> BOUNDARY DETECTED! Phrase sliced and saved to Dictionary.")
            
        # Simulate real-time delay between notes played by a musician
        time.sleep(duration / speed_multiplier)

    # Force a final boundary check for any remaining notes in the buffer
    time.sleep(1.0 / speed_multiplier)
    builder.add_note(0)  # Dummy silent note to flush buffer

    print("\n--- Final Generated Phrase Dictionary ---")
    if not builder.dictionary:
        print("No phrases were captured. Try adjusting segmentation thresholds.")
    else:
        for length, phrases in builder.dictionary.items():
            print(f"Phrase Length ({length} notes):")
            for idx, phrase in enumerate(phrases, 1):
                print(f"  [{idx}] MIDI Sequence: {phrase}")
    print("-" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Melodict phrase segmentation.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--threshold", type=float, default=1.5, help="LBDM sensitivity threshold.")
    args = parser.parse_args()

    run_simulation(args.speed, args.threshold)