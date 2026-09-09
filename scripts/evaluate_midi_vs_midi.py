"""
Offline Polyphonic Evaluation Script (MIDI vs MIDI)
Compares acausal/offline predictions against Ground Truth MIDI.
Outputs standard MIR multipitch metrics to the console and exports them to a CSV.
"""
import argparse
import csv
import glob
import os
import sys

import librosa
import mir_eval
import numpy as np
import pretty_midi
from tqdm import tqdm


def get_active_pitches_at_times(midi_path: str, times: np.ndarray) -> list[np.ndarray]:
    """Parses a MIDI file and returns a list of active frequencies in Hz for each time step."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI {midi_path}: {e}")
        return [np.array([]) for _ in times]
        
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
        
    return freqs


def main():
    parser = argparse.ArgumentParser(description="Evaluate offline MIDI extractions against Ground Truth.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with Ground Truth (.midi) files")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory with predicted (.mid) files")
    parser.add_argument("--pred_suffix", type=str, default="", help="Suffix of the predicted files (leave empty if none)")
    parser.add_argument("--buffer_ms", type=int, default=46, help="Evaluation window size in ms (to match streaming eval)")
    parser.add_argument("--out_csv", type=str, default="exports/offline_evaluation.csv", help="Path to export the CSV results")
    args = parser.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.midi")) + glob.glob(os.path.join(args.gt_dir, "*.mid")))
    
    if not gt_files:
        print(f"No ground truth MIDI files found in {args.gt_dir}")
        sys.exit(1)

    all_scores = []
    hop_time = args.buffer_ms / 1000.0

    print(f"Comparing Ground Truth vs predictions in '{args.pred_dir}'...")
    
    for gt_path in tqdm(gt_files, desc="Evaluating pairs"):
        base_name = os.path.splitext(os.path.basename(gt_path))[0]
        # Dynamically build the predicted path based on suffix
        pred_path = os.path.join(args.pred_dir, f"{base_name}{args.pred_suffix}.mid")
        
        if not os.path.exists(pred_path):
            # Fallback: check if the extension was saved as .midi instead of .mid
            pred_path_fallback = os.path.join(args.pred_dir, f"{base_name}{args.pred_suffix}.midi")
            if os.path.exists(pred_path_fallback):
                pred_path = pred_path_fallback
            else:
                continue

        try:
            gt_midi = pretty_midi.PrettyMIDI(gt_path)
            duration = gt_midi.get_end_time()
        except Exception:
            continue
            
        times = np.arange(0, duration, hop_time)
        ref_freqs = get_active_pitches_at_times(gt_path, times)
        est_freqs = get_active_pitches_at_times(pred_path, times)
        
        try:
            scores = mir_eval.multipitch.evaluate(
                times, ref_freqs, 
                times, est_freqs
            )
            
            p = scores['Precision']
            r = scores['Recall']
            scores['F-measure'] = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            scores['Filename'] = base_name
            
            all_scores.append(scores)
        except ValueError as e:
            print(f"\nSkipping {base_name} due to evaluation error: {e}")

    if not all_scores:
        print("\nNo matching pairs found to evaluate. Check your directories and suffixes.")
        sys.exit(1)

    # Calculate averages
    avg_f1 = np.mean([s['F-measure'] for s in all_scores])
    avg_precision = np.mean([s['Precision'] for s in all_scores])
    avg_recall = np.mean([s['Recall'] for s in all_scores])
    avg_accuracy = np.mean([s['Accuracy'] for s in all_scores])

    # Export to CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, mode='w', newline='') as csv_file:
        fieldnames = ['Filename', 'F-measure', 'Precision', 'Recall', 'Accuracy']
        other_keys = [k for k in all_scores[0].keys() if k not in fieldnames]
        fieldnames.extend(other_keys)
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for score in all_scores:
            writer.writerow(score)
            
        avg_row = {k: '' for k in fieldnames}
        avg_row['Filename'] = 'AVERAGE'
        avg_row['F-measure'] = avg_f1
        avg_row['Precision'] = avg_precision
        avg_row['Recall'] = avg_recall
        avg_row['Accuracy'] = avg_accuracy
        writer.writerow(avg_row)

    print("\n" + "="*60)
    print("         POLYPHONIC EVALUATION RESULTS         ")
    print("="*60)
    print(f"Total pairs evaluated: {len(all_scores)}")
    print(f"Average F-Measure (F1): {avg_f1:.4f}")
    print(f"Average Precision:      {avg_precision:.4f}")
    print(f"Average Recall:         {avg_recall:.4f}")
    print(f"Average Accuracy:       {avg_accuracy:.4f}")
    print("="*60)
    print(f"Detailed results exported to: {args.out_csv}")


if __name__ == "__main__":
    main()