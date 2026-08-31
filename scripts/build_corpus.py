"""Extracts melodies from audio, segments them into phrases, and stores them in the PostgreSQL corpus."""

import argparse
from pathlib import Path

import librosa
from tqdm import tqdm

from melodict.database.models import Artist, Phrase, Session, SessionLocal, init_db
from melodict.extraction.sota_models import EngineFactory


def process_and_store(audio_path: Path, artist_name: str, duration: float | None = None, buffer_ms: int = 46) -> None:
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
            dataset_source=audio_path.name,
            bpm=120.0,
            key_signature="Unknown"
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)

        engine = EngineFactory.create("basic-pitch")
        if not engine:
            print("Error: Could not load basic-pitch.")
            return

        sample_rate = 44100
        hop_length = int(sample_rate * (buffer_ms / 1000.0))
        y, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration)
        
        MIN_PITCH = 50  # A2 - Cutoff for bass accompaniment bleeding

        raw_pitches = []
        for i in tqdm(range(0, len(y), hop_length), desc=f"Extracting {audio_path.name[:15]}..."):
            buffer = y[i : i + hop_length]
            if len(buffer) < hop_length:
                break  
            pred = engine.predict_frame(buffer, sample_rate)
            pred_pitch = int(pred) if pred else 0
            
            # Filter out low bass notes, treating them as silence
            if pred_pitch > 0 and pred_pitch < MIN_PITCH:
                pred_pitch = 0
                
            raw_pitches.append(pred_pitch)

        smoothed_pitches = []
        for i in range(len(raw_pitches)):
            if i == 0 or i == len(raw_pitches) - 1:
                smoothed_pitches.append(raw_pitches[i])
            else:
                window = sorted([raw_pitches[i-1], raw_pitches[i], raw_pitches[i+1]])
                smoothed_pitches.append(window[1])

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
    parser = argparse.ArgumentParser(description="Build melody corpus from audio files or directories.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to a .wav file or directory of .wav files.")
    parser.add_argument("--artist", type=str, default="MAESTRO_Generic", help="Artist name for the DB.")
    parser.add_argument("--duration", type=float, default=None, help="Process only the first N seconds per file.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Path not found: {input_path}")
        return

    init_db()

    if input_path.is_file() and input_path.suffix == ".wav":
        files_to_process = [input_path]
    elif input_path.is_dir():
        files_to_process = list(input_path.rglob("*.wav"))
    else:
        print("Invalid path or no .wav files found.")
        return
        
    print(f"\nFound {len(files_to_process)} audio files. Starting batch processing...\n")
    
    for file_path in files_to_process:
        process_and_store(file_path, args.artist, args.duration)


if __name__ == "__main__":
    main()