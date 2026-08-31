"""Query random phrases from a specific artist in the database, compile into a MIDI, and plot a piano roll."""

import argparse
import os
import random

import matplotlib.pyplot as plt
import mido

from melodict.database.models import Artist, Phrase, Session as DbSession, SessionLocal


def export_compiled_phrases(artist_name: str, limit: int = 10, separation_ms: int = 500) -> None:
    with SessionLocal() as db:
        # Join tables to filter phrases by the exact artist name
        query = (
            db.query(Phrase)
            .join(Phrase.session)
            .join(DbSession.artist)
            .filter(Artist.name == artist_name)
        )
        
        all_phrases = query.all()
        
        if not all_phrases:
            print(f"No phrases found for artist: '{artist_name}'.")
            return

        sampled_phrases = random.sample(all_phrases, min(limit, len(all_phrases)))
        print(f"\nFound {len(all_phrases)} phrases for {artist_name}. Compiling {len(sampled_phrases)} into MIDI and Plot...\n")
        
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        
        mid = mido.MidiFile(type=0)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        ticks_per_beat = mid.ticks_per_beat
        def ms_to_ticks(ms: float) -> int:
            return int((ms / 500.0) * ticks_per_beat)

        track.append(mido.Message('program_change', program=0, time=0))

        next_wait_ms = 0
        plot_notes = []
        phrase_boundaries = []
        current_time_sec = 0.0
        
        for i, phrase_record in enumerate(sampled_phrases, 1):
            print(f"Adding Phrase {i} (ID: {phrase_record.id}): {phrase_record.note_count} notes")
            
            sequence = phrase_record.sequence
            if not sequence:
                continue
                
            merged = []
            c_pitch, c_dur, c_vel = sequence[0]
            for p, d, v in sequence[1:]:
                if p == c_pitch:
                    c_dur += d
                else:
                    merged.append((c_pitch, c_dur, c_vel))
                    c_pitch, c_dur, c_vel = p, d, v
            merged.append((c_pitch, c_dur, c_vel))
            
            for p, d, v in merged:
                dur_sec = d / 1000.0
                if p == 0:
                    next_wait_ms += d
                    current_time_sec += dur_sec
                else:
                    track.append(mido.Message('note_on', note=p, velocity=v, time=ms_to_ticks(next_wait_ms)))
                    track.append(mido.Message('note_off', note=p, velocity=0, time=ms_to_ticks(d)))
                    next_wait_ms = 0
                    plot_notes.append((current_time_sec, dur_sec, p))
                    current_time_sec += dur_sec
                    
            next_wait_ms += separation_ms
            current_time_sec += (separation_ms / 1000.0)
            phrase_boundaries.append(current_time_sec)
            
        midi_path = os.path.join(export_dir, f"corpus_audit_{artist_name}.mid")
        mid.save(midi_path)
        
        plt.figure(figsize=(15, 5))
        for start, dur, pitch in plot_notes:
            plt.barh(pitch, dur, left=start, height=0.8, color='#6C4AB6', align='center')
            
        for boundary in phrase_boundaries[:-1]:  
            plt.axvline(x=boundary - (separation_ms / 2000.0), color='#E94560', linestyle='--', linewidth=1.5, alpha=0.8)
            
        plt.xlabel("Time (seconds)", fontsize=12, fontweight='bold')
        plt.ylabel("MIDI Pitch", fontsize=12, fontweight='bold')
        plt.title(f"LBDM Segmented Phrases - {artist_name}", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle=':')
        
        plot_path = os.path.join(export_dir, f"corpus_audit_{artist_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSuccess! Compiled MIDI saved to: {midi_path}")
        print(f"Success! Piano Roll plot saved to: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit phrases for a specific artist.")
    parser.add_argument("--artist", type=str, default="MAESTRO_Generic_Midi", help="The artist name to query from the database.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of phrases to export.")
    args = parser.parse_args()

    export_compiled_phrases(args.artist, args.limit)


if __name__ == "__main__":
    main()