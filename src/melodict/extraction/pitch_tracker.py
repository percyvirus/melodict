"""Real-time acoustic and neural pitch tracking with RMS velocity extraction."""

import logging
import time
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("MelodictTracker")

class PitchTracker:
    """Extracts quantized (Pitch, Duration, Velocity) tuples from raw audio buffers."""

    def __init__(self, sample_rate: int = 44100, frame_size: int = 2048):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.last_note_time = time.time()
        self.current_pitch: Optional[int] = None

    @staticmethod
    def quantize_duration(duration_ms: float) -> int:
        """Quantize raw milliseconds into standard rhythmic bins (~120 BPM)."""
        if duration_ms < 160: return 125    # 16th note
        if duration_ms < 350: return 250    # 8th note
        if duration_ms < 700: return 500    # Quarter note
        if duration_ms < 1400: return 1000  # Half note
        return 2000                         # Whole note

    @staticmethod
    def quantize_velocity(velocity: int) -> int:
        """Quantize raw MIDI velocity into 5 expressive dynamic tiers."""
        if velocity < 30: return 25         # pp
        if velocity < 55: return 50         # p
        if velocity < 85: return 75         # mf
        if velocity < 110: return 100       # f
        return 127                          # ff

    def process_buffer(self, audio_buffer: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Analyze PCM buffer and return a quantized (pitch, duration, velocity) tuple if note changes."""
        # 1. Calculate RMS Energy for Velocity
        rms = np.sqrt(np.mean(audio_buffer ** 2))
        # Reduce scaling so normal room noise stays near 0
        raw_velocity = int(np.clip((rms - 0.015) * 800, 0, 127))

        # Raise silence threshold to prevent background room noise from triggering false notes
        if raw_velocity < 25 or rms < 0.02:
            return None

        # 2. Extract Pitch (Using Yin / Autocorrelation / Essentia simulation for rapid inference)
        # Note: Replace with SOTA neural tracker (CREPE/basic-pitch) when GPU inference is active
        zero_crossings = np.where(np.diff(np.signbit(audio_buffer)))[0]
        if len(zero_crossings) < 2:
            return None
            
        est_freq = self.sample_rate / (2 * np.mean(np.diff(zero_crossings)))
        if not (27.5 <= est_freq <= 4186.0):  # Piano compass range (A0 to C8)
            return None

        raw_pitch = int(round(69 + 12 * np.log2(est_freq / 440.0)))
        
        # 3. Detect onset / note change and calculate duration
        now = time.time()
        if raw_pitch != self.current_pitch:
            duration_ms = (now - self.last_note_time) * 1000.0
            self.last_note_time = now
            self.current_pitch = raw_pitch

            q_dur = self.quantize_duration(duration_ms)
            q_vel = self.quantize_velocity(raw_velocity)
            return (raw_pitch, q_dur, q_vel)

        return None