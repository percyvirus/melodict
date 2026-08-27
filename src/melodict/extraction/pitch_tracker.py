"""Real-time acoustic and neural pitch tracking with RMS velocity extraction."""

import logging
import time

import numpy as np

logger = logging.getLogger("MelodictTracker")

NoteTuple = tuple[int, int, int]


class PitchTracker:
    """Extracts quantized (Pitch, Duration, Velocity) tuples from raw audio buffers."""

    def __init__(self, sample_rate: int = 44100, frame_size: int = 2048) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.last_note_time = time.time()
        self.current_pitch: int | None = None

    @staticmethod
    def quantize_duration(duration_ms: float) -> int:
        """Quantize raw milliseconds into standard rhythmic bins (~120 BPM)."""
        if duration_ms < 160: return 125    
        if duration_ms < 350: return 250    
        if duration_ms < 700: return 500    
        if duration_ms < 1400: return 1000  
        return 2000                         

    @staticmethod
    def quantize_velocity(velocity: int) -> int:
        """Quantize raw MIDI velocity into 5 expressive dynamic tiers."""
        if velocity < 30: return 25         
        if velocity < 55: return 50         
        if velocity < 85: return 75         
        if velocity < 110: return 100       
        return 127                          

    def process_buffer(self, audio_buffer: np.ndarray) -> NoteTuple | None:
        """Analyze PCM buffer and return a quantized (pitch, duration, velocity) tuple if note changes."""
        rms = np.sqrt(np.mean(audio_buffer ** 2))
        raw_velocity = int(np.clip((rms - 0.015) * 800, 0, 127))

        if raw_velocity < 25 or rms < 0.02:
            return None

        zero_crossings = np.where(np.diff(np.signbit(audio_buffer)))[0]
        if len(zero_crossings) < 2:
            return None
            
        est_freq = self.sample_rate / (2 * np.mean(np.diff(zero_crossings)))
        if not (27.5 <= est_freq <= 4186.0):
            return None

        raw_pitch = round(69 + 12 * np.log2(est_freq / 440.0))
        
        now = time.time()
        if raw_pitch != self.current_pitch:
            duration_ms = (now - self.last_note_time) * 1000.0
            self.last_note_time = now
            self.current_pitch = raw_pitch

            q_dur = self.quantize_duration(duration_ms)
            q_vel = self.quantize_velocity(raw_velocity)
            return (raw_pitch, q_dur, q_vel)

        return None