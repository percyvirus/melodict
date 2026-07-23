"""Real-time pitch tracking bridge connecting incoming OSC audio buffers to SOTA extraction engines."""

import logging
from typing import Any, List, Optional
import numpy as np

from melodict.extraction.sota_models import EngineFactory, PitchExtractorEngine

logger = logging.getLogger("MelodictPitchTracker")


class PitchTracker:
    """Manages audio buffer conversion from OSC network packets and executes symbolic pitch prediction."""

    def __init__(self, engine_name: str = "essentia-yin") -> None:
        """
        Initialize the pitch tracker with a specific SOTA engine.
        
        Args:
            engine_name: The target engine ('essentia-yin', 'crepe', 'pyin', or 'basic-pitch').
        """
        self.engine_name = engine_name
        self.engine: Optional[PitchExtractorEngine] = EngineFactory.create(engine_name)
        if not self.engine:
            logger.error(f"Failed to load engine '{engine_name}'. Falling back to pyin.")
            self.engine = EngineFactory.create("pyin")

    def estimate_pitch(self, audio_args: List[Any], sample_rate: int = 44100) -> Optional[int]:
        """
        Convert raw OSC float lists from Max/MSP into PCM arrays and predict the MIDI note.
        
        Args:
            audio_args: Sequence of float values received from the UDP network bundle.
            sample_rate: Audio sampling rate matching the Max/MSP DSP settings.
            
        Returns:
            Estimated symbolic MIDI note integer, or None if unvoiced/silence.
        """
        if not audio_args or not self.engine:
            return None

        try:
            # Convert OSC network float sequence to a standard 1D NumPy PCM buffer
            audio_buffer = np.array(audio_args, dtype=np.float32)
            
            # Check for absolute silence or flatline signal to save CPU cycles
            if np.max(np.abs(audio_buffer)) < 0.01:
                return None
                
            return self.engine.predict_frame(audio_buffer, sample_rate=sample_rate)
        except Exception as e:
            logger.error(f"Error during live pitch estimation: {e}")
            return None