"""OSC server and client bridge for real-time bidirectional communication with Max/MSP."""

import argparse
import logging
import time
from melodict.utils.session_recorder import SessionRecorder
from typing import Any, List
from pythonosc import dispatcher, osc_server, udp_client

from melodict.extraction.pitch_tracker import PitchTracker
from melodict.segmentation.phrase_builder import PhraseBuilder
from melodict.generation.continuator import VMMContinuator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MelodictOSC")


class OSCBridge:
    """Manages UDP listening from Max/MSP, live phrase segmentation, and reactive AI continuation."""

    def __init__(self, receive_ip: str, receive_port: int, send_ip: str, send_port: int) -> None:
        self.receive_ip = receive_ip
        self.receive_port = receive_port
        self.client = udp_client.SimpleUDPClient(send_ip, send_port)
        
        # Initialize core musical processing pipeline
        self.pitch_tracker = PitchTracker()
        self.phrase_builder = PhraseBuilder(boundary_threshold=1.5, max_silence_sec=0.8)
        self.continuator = VMMContinuator(max_order=4)
        
        # Setup OSC message dispatcher
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/audio/features", self._handle_audio_features)
        self.dispatcher.map("/midi/note_in", self._handle_midi_in)
        self.dispatcher.set_default_handler(self._default_handler)
        
        self.recorder = SessionRecorder(output_dir="recordings")

    def _handle_audio_features(self, address: str, *args: List[Any]) -> None:
        """Process incoming raw audio feature frames from Max/MSP."""
        if not args:
            return
        
        midi_note = self.pitch_tracker.estimate_pitch(args)
        if midi_note is not None:
            self._process_symbolic_note(midi_note)

    def _handle_midi_in(self, address: str, *args: List[Any]) -> None:
        """Process incoming symbolic MIDI keyboard data."""
        if len(args) >= 2:
            note, velocity = int(args[0]), int(args[1])
            if velocity > 0:
                self._process_symbolic_note(note)

    def _process_symbolic_note(self, note: int) -> None:
        """Feed notes to the segmentation engine and trigger AI continuation on boundaries."""
        # Send cleaned symbolic note back to Max for real-time monitoring
        self.client.send_message("/midi/note_out", note)
        
        # Check if this note triggered the end of a phrase
        is_boundary = self.phrase_builder.add_note(note)
        
        if is_boundary and self.phrase_builder.dictionary:
            logger.info("Phrase boundary detected! Triggering Continuator response...")
            
            # 1. Update the VMM tree with all recently captured phrases
            self.continuator.learn_from_dictionary(self.phrase_builder.dictionary)
            
            # 2. Retrieve the last completed phrase played by the musician
            last_length = list(self.phrase_builder.dictionary.keys())[-1]
            last_phrase = self.phrase_builder.dictionary[last_length][-1]
            
            # 3. Generate an immediate stylistic response of similar length
            response_phrase = self.continuator.generate_continuation(
                input_phrase=last_phrase, target_length=len(last_phrase), temperature=0.7
            )
            
            logger.info(f"Musician played: {last_phrase} -> AI Answer: {response_phrase}")
            
            self.recorder.log_phrase("Melodict_AI", response_phrase)
            
            # 4. Dispatch the generated response sequence to Max/MSP
            # Sending as a list to let Max/MSP sequence the rhythm, or trigger individual notes
            for idx, resp_note in enumerate(response_phrase):
                self.client.send_message("/midi/ai_answer", [resp_note, idx])

    def _default_handler(self, address: str, *args: List[Any]) -> None:
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