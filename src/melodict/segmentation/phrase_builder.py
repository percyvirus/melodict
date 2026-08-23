"""Gestalt-based Local Boundary Detection Model (LBDM) for musical phrase segmentation."""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("MelodictLBDM")

# Type alias for our 3-variable musical state
NoteTuple = Tuple[int, int, int]  # (Pitch, Duration, Velocity)

class PhraseBuilder:
    """Segments continuous note streams into phrases and builds an on-the-fly dictionary."""

    def __init__(self, max_phrase_length: int = 12, silence_threshold_ms: int = 550):
        self.max_phrase_length = max_phrase_length
        self.silence_threshold_ms = silence_threshold_ms
        self.current_phrase: List[NoteTuple] = []
        # Dictionary mapping phrase lengths to lists of recorded phrases
        self.dictionary: Dict[int, List[List[NoteTuple]]] = {}

    def add_note(self, note_event: NoteTuple) -> bool:
        """Add a note event tuple and evaluate LBDM boundary rules. Returns True if segmented."""
        pitch, duration, velocity = note_event
        self.current_phrase.append(note_event)

        is_boundary = False

        # Rule 1: Temporal rest exceeds threshold
        if duration >= self.silence_threshold_ms:
            is_boundary = True
        # Rule 2: Large melodic interval leap (> Major 6th / 9 semitones)
        elif len(self.current_phrase) >= 2:
            prev_pitch = self.current_phrase[-2][0]
            if abs(pitch - prev_pitch) >= 9:
                is_boundary = True
        # Rule 3: Safety buffer limit
        if len(self.current_phrase) >= self.max_phrase_length:
            is_boundary = True

        if is_boundary and len(self.current_phrase) >= 2:
            phrase_len = len(self.current_phrase)
            if phrase_len not in self.dictionary:
                self.dictionary[phrase_len] = []
            self.dictionary[phrase_len].append(self.current_phrase.copy())
            
            logger.info(f"Segmented phrase of length {phrase_len}: {self.current_phrase}")
            self.current_phrase.clear()
            return True

        return False