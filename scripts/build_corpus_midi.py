"""Extracts monophonic melodies from polyphonic MIDI files, segments them, and stores them in the DB."""

import argparse
from pathlib import Path

import mido
from tqdm import tqdm

from melodict.database.models import Artist, Phrase, Session, SessionLocal, init_db


def process_and_store_midi(midi_path: Path, artist_name: str, duration: float | None = None, buffer_ms: int = 46) -> None:
    with SessionLocal() as db:
        artist = db.query(Artist).filter_by(name=artist_name).first()
        if not artist:
            artist = Artist(name=artist_name, genre="Classical", default_instrument="Piano")
            db.add(artist)
            db.commit()
            db.refresh(artist)

        db_session = Session(
            artist_id=artist.id,
            instrument="Piano",
            dataset_source=midi_path.name,
            bpm=120.0,
            key_signature="Unknown"
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)

        # 1. Parse MIDI into absolute time events
        mid = mido.MidiFile(midi_path)
        events = []
        current_time_s = 0.0
        
        for msg in mid:
            current_time_s += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append((current_time_s, 'on', msg.note))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                events.append((current_time_s, 'off', msg.note))
                
        # 2. Simulate 46ms framing and extract Skyline (highest pitch)
        hop_s = buffer_ms / 1000.0
        target_duration = current_time_s if duration is None else min(duration, current_time_s)
        
        raw_pitches = []
        time_cursor = 0.0
        event_idx = 0
        active_notes = set()
        MIN_PITCH = 45  # A2 cutoff
        
        while time_cursor < target_duration:
            while event_idx < len(events) and events[event_idx][0] <= time_cursor:
                _, ev_type, ev_note = events[event_idx]
                if ev_type == 'on':
                    active_notes.add(ev_note)
                else:
                    active_notes.discard(ev_note)
                event_idx += 1
                
            highest_pitch = max(active_notes) if active_notes else 0
            if highest_pitch < MIN_PITCH:
                highest_pitch = 0
            raw_pitches.append(highest_pitch)
            time_cursor += hop_s

        # 3. Median Filter (to clean microscopic overlaps or trills)
        smoothed_pitches = []
        for i in range(len(raw_pitches)):
            if i == 0 or i == len(raw_pitches) - 1:
                smoothed_pitches.append(raw_pitches[i])
            else:
                window = sorted([raw_pitches[i-1], raw_pitches[i], raw_pitches[i+1]])
                smoothed_pitches.append(window[1])

        # 4. State machine: consolidate notes, filter blips, enforce Gestalt cuts
        current_sequence = []
        active_pitch = 0
        active_duration = 0
        last_real_pitch = 0
        
        dictionary = {}

        MIN_NOTE_DURATION = 92  
        MAX_INTERVAL = 12       
        MIN_PHRASE_NOTES = 5    
        MAX_REST_MS = 550

        def save_phrase():
            while current_sequence and current_sequence[-1][0] == 0:
                current_sequence.pop()
            while current_sequence and current_sequence[0][0] == 0:
                current_sequence.pop(0)
                
            real_notes = [n for n in current_sequence if n[0] > 0]
            if len(real_notes) >= MIN_PHRASE_NOTES:
                dictionary.setdefault(len(real_notes), []).append(current_sequence.copy())
            
            current_sequence.clear()
            return 0

        for pitch in smoothed_pitches:
            if pitch == active_pitch:
                active_duration += buffer_ms
            else:
                if active_pitch > 0:
                    if active_duration >= MIN_NOTE_DURATION:
                        if last_real_pitch > 0 and abs(active_pitch - last_real_pitch) > MAX_INTERVAL:
                            last_real_pitch = save_phrase()
                            
                        current_sequence.append((active_pitch, active_duration, 100))
                        last_real_pitch = active_pitch
                    else:
                        if current_sequence:
                            if current_sequence[-1][0] == 0:
                                current_sequence[-1] = (0, current_sequence[-1][1] + active_duration, 0)
                            else:
                                current_sequence.append((0, active_duration, 0))
                else:
                    if active_duration >= MAX_REST_MS:
                        last_real_pitch = save_phrase()
                    elif active_duration > 0 and current_sequence:
                        if current_sequence[-1][0] == 0:
                            current_sequence[-1] = (0, current_sequence[-1][1] + active_duration, 0)
                        else:
                            current_sequence.append((0, active_duration, 0))
                            
                active_pitch = pitch
                active_duration = buffer_ms

        if active_pitch > 0 and active_duration >= MIN_NOTE_DURATION:
            if last_real_pitch > 0 and abs(active_pitch - last_real_pitch) > MAX_INTERVAL:
                last_real_pitch = save_phrase()
            current_sequence.append((active_pitch, active_duration, 100))
        save_phrase()

        phrase_count = 0
        for length, phrases in dictionary.items():
            for sequence in phrases:
                duration_ms = sum(note[1] for note in sequence)
                db_phrase = Phrase(
                    session_id=db_session.id,
                    note_count=length,
                    duration_ms=duration_ms,
                    sequence=sequence
                )
                db.add(db_phrase)
                phrase_count += 1
                
        db.commit()
        print(f"Saved {phrase_count} high-quality phrases to DB (Session ID {db_session.id}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build melody corpus directly from MIDI files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to a .mid file or directory of .mid files.")
    parser.add_argument("--artist", type=str, default="MAESTRO_Generic_Midi", help="Artist name for the DB.")
    parser.add_argument("--duration", type=float, default=None, help="Process only the first N seconds per file.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Path not found: {input_path}")
        return

    init_db()

    if input_path.is_file() and (input_path.suffix.lower() == ".mid" or input_path.suffix.lower() == ".midi"):
        files_to_process = [input_path]
    elif input_path.is_dir():
        files_to_process = list(input_path.rglob("*.mid")) + list(input_path.rglob("*.midi"))
    else:
        print("Invalid path or no .mid files found.")
        return
        
    print(f"\nFound {len(files_to_process)} MIDI files. Starting batch processing...\n")
    
    for file_path in tqdm(files_to_process, desc="Batch Progress"):
        process_and_store_midi(file_path, args.artist, args.duration)


if __name__ == "__main__":
    main()