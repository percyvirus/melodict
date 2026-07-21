"""OSC server and client bridge for real-time bidirectional communication with Max/MSP."""

import argparse
import logging
from typing import Any, List
from pythonosc import dispatcher, osc_server, udp_client

from melodict.extraction.pitch_tracker import PitchTracker
from melodict.segmentation.phrase_builder import PhraseBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MelodictOSC")


class OSCBridge:
    """Manages UDP listening from Max/MSP and sending processed symbolic MIDI data back."""

    def __init__(self, receive_ip: str, receive_port: int, send_ip: str, send_port: int) -> None:
        self.receive_ip = receive_ip
        self.receive_port = receive_port
        self.client = udp_client.SimpleUDPClient(send_ip, send_port)
        
        # Initialize musical processing modules
        self.pitch_tracker = PitchTracker()
        self.phrase_builder = PhraseBuilder()
        
        # Setup OSC message dispatcher
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/audio/features", self._handle_audio_features)
        self.dispatcher.map("/midi/note_in", self._handle_midi_in)
        self.dispatcher.set_default_handler(self._default_handler)

    def _handle_audio_features(self, address: str, *args: List[Any]) -> None:
        """Process incoming raw audio feature frames from Max/MSP."""
        if not args:
            return
        
        # Estimate MIDI pitch from feature frame
        midi_note = self.pitch_tracker.estimate_pitch(args)
        if midi_note is not None:
            logger.info(f"Detected Pitch: {midi_note} -> Sending to Max")
            self.client.send_message("/midi/note_out", midi_note)
            self.phrase_builder.add_note(midi_note)

    def _handle_midi_in(self, address: str, *args: List[Any]) -> None:
        """Process incoming symbolic MIDI data for real-time phrase segmentation."""
        if len(args) >= 2:
            note, velocity = int(args[0]), int(args[1])
            if velocity > 0:
                is_boundary = self.phrase_builder.add_note(note)
                if is_boundary:
                    logger.info("Phrase boundary detected! Triggering dictionary update.")
                    self.client.send_message("/phrase/boundary", 1)

    def _default_handler(self, address: str, *args: List[Any]) -> None:
        """Fallback handler for unmapped OSC messages."""
        logger.debug(f"Received unmapped OSC message: {address}: {args}")

    def start(self) -> None:
        """Start the blocking OSC server loop."""
        server = osc_server.ThreadingOSCUDPServer(
            (self.receive_ip, self.receive_port), self.dispatcher
        )
        logger.info(f"Melodict OSC Server listening on {self.receive_ip}:{self.receive_port}")
        logger.info(f"Sending predictions to Max/MSP on port {self.client._port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("OSC Server stopped by user.")
        finally:
            server.server_close()


def main() -> None:
    """Command-line entry point for running the Melodict OSC bridge."""
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