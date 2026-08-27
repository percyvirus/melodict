"""OSC server and client bridge for real-time bidirectional communication with Max/MSP."""

import argparse
import logging
import threading
import time
from typing import Any

import numpy as np
from pythonosc import dispatcher, osc_server, udp_client

from melodict.extraction.pitch_tracker import PitchTracker
from melodict.generation.continuator import VMMContinuator
from melodict.segmentation.phrase_builder import PhraseBuilder
from melodict.utils.session_recorder import SessionRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MelodictOSC")

NoteTuple = tuple[int, int, int]  # (Pitch, Duration, Velocity)


class OSCBridge:
    """Manages UDP listening from Max/MSP, live phrase segmentation, and reactive AI continuation."""

    def __init__(self, receive_ip: str, receive_port: int, send_ip: str, send_port: int) -> None:
        self.receive_ip = receive_ip
        self.receive_port = receive_port
        self.client = udp_client.SimpleUDPClient(send_ip, send_port)
        
        self.pitch_tracker = PitchTracker()
        self.phrase_builder = PhraseBuilder(max_phrase_length=12, silence_threshold_ms=550)
        self.continuator = VMMContinuator(max_order=3)
        
        self.last_midi_time = time.time()
        
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/audio/features", self._handle_audio_features)
        self.dispatcher.map("/midi/note_in", self._handle_midi_in)
        self.dispatcher.set_default_handler(self._default_handler)
        
        self.recorder = SessionRecorder(output_dir="recordings")

    def _handle_audio_features(self, address: str, *args: Any) -> None:
        """Process incoming raw PCM audio buffers from Max/MSP."""
        if not args:
            return
        
        audio_buffer = np.array(args, dtype=np.float32)
        note_event: NoteTuple | None = self.pitch_tracker.process_buffer(audio_buffer)
        
        if note_event is not None:
            self._process_symbolic_note(note_event)

    def _handle_midi_in(self, address: str, *args: Any) -> None:
        """Process incoming symbolic MIDI keyboard data and quantize duration/velocity."""
        if len(args) >= 2:
            pitch, velocity = int(args[0]), int(args[1])
            if velocity > 0:
                now = time.time()
                duration_ms = (now - self.last_midi_time) * 1000.0
                self.last_midi_time = now
                
                q_dur = PitchTracker.quantize_duration(duration_ms)
                q_vel = PitchTracker.quantize_velocity(velocity)
                
                self._process_symbolic_note((pitch, q_dur, q_vel))

    def _process_symbolic_note(self, note_event: NoteTuple) -> None:
        """Feed note tuples to the segmentation engine and trigger AI continuation on boundaries."""
        # Fix RUF059: Ignore unpacked variables that are not used locally
        pitch, _duration, _velocity = note_event
        
        self.client.send_message("/midi/note_out", pitch)
        
        is_boundary = self.phrase_builder.add_note(note_event)
        
        if is_boundary and self.phrase_builder.dictionary:
            logger.info("Phrase boundary detected! Triggering Continuator response...")
            
            self.continuator.learn_from_dictionary(self.phrase_builder.dictionary)
            
            last_length = list(self.phrase_builder.dictionary.keys())[-1]
            last_phrase = self.phrase_builder.dictionary[last_length][-1]
            
            response_phrase = self.continuator.generate_continuation(
                input_phrase=last_phrase, target_length=len(last_phrase), temperature=0.7
            )
            
            logger.info(f"Musician played: {last_phrase} -> AI Answer: {response_phrase}")
            
            def send_sequenced_response(phrase: list[NoteTuple]) -> None:
                for idx, (resp_pitch, resp_dur, resp_vel) in enumerate(phrase):
                    self.client.send_message("/midi/ai_answer", [resp_pitch, resp_dur, resp_vel, idx])
                    time.sleep(resp_dur / 1000.0)

            threading.Thread(target=send_sequenced_response, args=(response_phrase,), daemon=True).start()

    def _default_handler(self, address: str, *args: Any) -> None:
        logger.debug(f"Received unmapped OSC message: {address}: {args}")

    def start(self) -> None:
        server = osc_server.ThreadingOSCUDPServer(
            (self.receive_ip, self.receive_port), self.dispatcher
        )
        logger.info(f"Melodict OSC Server listening on {self.receive_ip}:{self.receive_port}")
        logger.info(f"Sending AI continuations to Max/MSP on port {self.client._port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("OSC Server stopped by user.")
        finally:
            server.server_close()
            self.recorder.export_midi_file("session_export.mid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Melodict OSC Bridge Server.")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to listen on.")
    parser.add_argument("--port", type=int, default=8001, help="UDP port to listen for Max/MSP.")
    parser.add_argument("--send_ip", type=str, default="127.0.0.1", help="Target IP for Max/MSP.")
    parser.add_argument("--send_port", type=int, default=8000, help="Target UDP port for Max/MSP.")
    args = parser.parse_args()

    bridge = OSCBridge(args.ip, args.port, args.send_ip, args.send_port)
    bridge.start()


if __name__ == "__main__":
    main()