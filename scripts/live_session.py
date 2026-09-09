"""
Melodict Live Session Orchestrator
Captures live audio via BlackHole, performs real-time extraction, applies LBDM segmentation,
and triggers the Variable Markov Model for musical generation.
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np

try:
    import sounddevice as sd
    from pythonosc import udp_client
except ImportError:
    print("Missing dependencies. Run: uv add sounddevice python-osc numpy sqlalchemy")
    sys.exit(1)

# Ensure src/ is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.melodict.database.models import Artist, Phrase, SessionLocal
from src.melodict.database.models import Session as DBSessionModel
from src.melodict.extraction.sota_models import EngineFactory
from src.melodict.generation.continuator import VMMContinuator
from src.melodict.segmentation.phrase_builder import PhraseBuilder

# Global audio queue for thread-safe processing
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Reads audio from BlackHole and pushes it to the processing queue."""
    if status:
        print(f"Audio Status: {status}", file=sys.stderr)
    mono_data = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
    audio_queue.put(mono_data.copy())

def parse_arguments():
    parser = argparse.ArgumentParser(description="Melodict Live Session Orchestrator")
    parser.add_argument("--filter-artists", type=str, default=None, 
                        help="Comma-separated list of artists (e.g., 'Ignasi, Bill Evans')")
    parser.add_argument("--filter-sessions", type=str, default=None, 
                        help="Comma-separated list of sessions (e.g., 'Session_1')")
    parser.add_argument("--learn-as", type=str, default=None,
                        help="Save new live phrases to this session name in Postgres")
    parser.add_argument("--device", type=str, default="BlackHole", help="Input device name")
    parser.add_argument("--buffer_ms", type=int, default=46, help="Buffer size in ms")
    parser.add_argument("--osc_ip", type=str, default="127.0.0.1", help="Max8 IP")
    parser.add_argument("--osc_port", type=int, default=8000, help="Max8 UDP port")
    return parser.parse_args()

def fetch_corpus(db_session, artists_arg, sessions_arg):
    """Fetches symbolic phrases from PostgreSQL based on CLI filters."""
    query = db_session.query(Phrase)
    if artists_arg:
        artist_names = [a.strip() for a in artists_arg.split(",")]
        query = query.join(DBSessionModel).join(Artist).filter(Artist.name.in_(artist_names))
    if sessions_arg:
        session_names = [s.strip() for s in sessions_arg.split(",")]
        # Only join DBSessionModel if it wasn't joined by the artist filter
        if not artists_arg:
            query = query.join(DBSessionModel)
        query = query.filter(DBSessionModel.dataset_source.in_(session_names))
    return query.all()

def get_or_create_session(db_session, session_name):
    """Ensures the target session exists for learning mode."""
    session_record = db_session.query(DBSessionModel).filter_by(dataset_source=session_name).first()
    if not session_record:
        # Create a dummy artist for live recordings if it doesn't exist
        artist = db_session.query(Artist).filter_by(name="LiveUser").first()
        if not artist:
            artist = Artist(name="LiveUser", genre="Live", default_instrument="Piano")
            db_session.add(artist)
            db_session.commit()
            
        session_record = DBSessionModel(artist_id=artist.id, dataset_source=session_name, instrument="Piano")
        db_session.add(session_record)
        db_session.commit()
    return session_record

def main():
    args = parse_arguments()
    
    # 1. Initialize OSC Client
    osc_client = udp_client.SimpleUDPClient(args.osc_ip, args.osc_port)
    print(f"[OSC] Bridge initialized on {args.osc_ip}:{args.osc_port}")

    # 2. Database Connection & Corpus
    db_session = SessionLocal()
    corpus = fetch_corpus(db_session, args.filter_artists, args.filter_sessions)
    
    # Pre-load corpus into dictionary structure for VMM
    corpus_dict = {}
    for phrase in corpus:
        # phrase.sequence is stored as JSON list of lists: [[pitch, dur, vel], ...]
        seq = [tuple(n) for n in phrase.sequence]
        l = len(seq)
        corpus_dict.setdefault(l, []).append(seq)
    
    # 3. Initialize Engines
    extractor = EngineFactory.create("basic-pitch")
    if not extractor:
        print("Error: Could not load basic-pitch engine.")
        return
        
    segmenter = PhraseBuilder(max_phrase_length=12, silence_threshold_ms=550)
    markov_model = VMMContinuator(max_order=3)
    markov_model.learn_from_dictionary(corpus_dict)
    
    print(f"[INIT] Loaded {len(corpus)} phrases from DB. Learning mode: {'Enabled (' + args.learn_as + ')' if args.learn_as else 'Disabled'}")
    
    active_session_record = get_or_create_session(db_session, args.learn_as) if args.learn_as else None

    # 4. Start Capture
    samplerate = 44100
    blocksize = int(samplerate * (args.buffer_ms / 1000.0))
    
    device_idx = None
    for idx, d in enumerate(sd.query_devices()):
        if args.device.lower() in d['name'].lower() and d['max_input_channels'] > 0:
            device_idx = idx
            break
            
    if device_idx is None:
        print(f"Error: Could not find audio device '{args.device}'.")
        return

    print(f"[AUDIO] Starting capture on: {sd.query_devices()[device_idx]['name']}")
    
    # State tracking to convert streaming frames into NoteTuples
    active_pitch = 0
    active_duration_ms = 0
    silence_duration_ms = 0
    
    try:
        with sd.InputStream(device=device_idx, channels=2, samplerate=samplerate, 
                            blocksize=blocksize, callback=audio_callback):
            print("\n[LIVE] Listening... Press Ctrl+C to stop.")
            while True:
                audio_frame = audio_queue.get()
                
                raw_pitch = extractor.predict_frame(audio_frame, sample_rate=samplerate)
                if raw_pitch is None:
                    raw_pitch = 0
                    
                if 0 < raw_pitch < 45: # Smart Skyline cutoff
                    raw_pitch = 0
                    
                # Note transition logic
                if raw_pitch != active_pitch:
                    if active_pitch > 0:
                        # Note ended, build tuple and add to segmenter
                        note_tuple = (active_pitch, active_duration_ms, 100)
                        
                        # Store reference before add_note clears it
                        completed_phrase = segmenter.current_phrase.copy() + [note_tuple]
                        
                        is_boundary = segmenter.add_note(note_tuple)
                        
                        if is_boundary:
                            print(f"\n[LBDM] Boundary detected! Phrase length: {len(completed_phrase)}")
                            
                            # Generation
                            response_sequence = markov_model.generate_continuation(completed_phrase)
                            
                            def play_melody(seq):
                                for n in seq:
                                    pitch, duration_ms, _velocity = n
                                    osc_client.send_message("/midi/ai_answer", [pitch, 0])
                                    time.sleep(duration_ms / 1000.0)
                            
                            # Lanzamos la melodía en segundo plano para no congelar el audio
                            threading.Thread(target=play_melody, args=(response_sequence,), daemon=True).start()
                            
                            print(f"[VMM] Generated {len(response_sequence)} response notes.")
                            
                            # Database Save
                            if active_session_record:
                                new_phrase = Phrase(
                                    session_id=active_session_record.id,
                                    note_count=len(completed_phrase),
                                    duration_ms=sum(d for p, d, v in completed_phrase),
                                    sequence=[[int(p), int(d), int(v)] for p, d, v in completed_phrase]
                                )
                                db_session.add(new_phrase)
                                db_session.commit()
                                print(f"[DB] Saved phrase to session '{args.learn_as}'")
                                
                    active_pitch = raw_pitch
                    active_duration_ms = int(args.buffer_ms)
                    silence_duration_ms = 0
                else:
                    if active_pitch > 0:
                        active_duration_ms += int(args.buffer_ms)
                    else:
                        silence_duration_ms += int(args.buffer_ms)
                        # Force boundary if silence exceeds threshold
                        if silence_duration_ms > segmenter.silence_threshold_ms and len(segmenter.current_phrase) > 0:
                            completed_phrase = segmenter.current_phrase.copy()
                            print(f"\n[LBDM] Acoustic silence boundary. Phrase length: {len(completed_phrase)}")
                            response_sequence = markov_model.generate_continuation(completed_phrase)
                            for i, note in enumerate(response_sequence):
                                osc_client.send_message("/midi/ai_answer", [note[0], i])
                            segmenter.current_phrase.clear()
                            silence_duration_ms = 0

    except KeyboardInterrupt:
        print("\n[EXIT] Closing Melodict Live Session.")
        db_session.close()

if __name__ == "__main__":
    main()