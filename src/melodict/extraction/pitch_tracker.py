"""Standard acoustic pitch tracking and MIDI estimation utilities."""

from typing import Any, List, Optional
import numpy as np


class PitchTracker:
    """Low-latency fundamental frequency (F0) to MIDI pitch tracker."""

    def __init__(self, min_freq: float = 60.0, max_freq: float = 2000.0) -> None:
        """
        Initialize the pitch tracker with frequency bounds.
        
        Args:
            min_freq: Minimum frequency in Hz (replaces the legacy 60Hz filter).
            max_freq: Maximum detectable frequency in Hz.
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.last_note: Optional[int] = None

    def freq_to_midi(self, freq: float) -> Optional[int]:
        """Convert frequency in Hertz to the nearest MIDI note number."""
        if freq < self.min_freq or freq > self.max_freq:
            return None
        midi_float = 69.0 + 12.0 * np.log2(freq / 440.0)
        return int(np.round(midi_float))

    def estimate_pitch(self, audio_features: List[Any]) -> Optional[int]:
        """
        Estimate MIDI pitch from incoming feature vectors (e.g., FFT bins or Aubio outputs).
        
        Args:
            audio_features: List containing numerical audio representations from OSC.
            
        Returns:
            Estimated MIDI note number or None if silence/unvoiced.
        """
        if not audio_features:
            return None

        # Extract the primary frequency (assuming first argument is fundamental frequency from Max)
        try:
            raw_val = float(audio_features[0])
            # If Max sends frequency directly (e.g., from aubiopitch~)
            if raw_val > 0:
                midi_note = self.freq_to_midi(raw_val)
                if midi_note != self.last_note:
                    self.last_note = midi_note
                    return midi_note
        except (ValueError, TypeError):
            return None
            
        return None