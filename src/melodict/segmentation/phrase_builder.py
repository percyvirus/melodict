"""Accumulates symbolic notes in real-time and builds structured phrase dictionaries."""

import time
from typing import Dict, List
from melodict.segmentation.lbdm import LBDM


class PhraseBuilder:
    """Manages real-time note streaming and triggers phrase dictionary updates."""

    def __init__(self, boundary_threshold: float = 1.5, max_silence_sec: float = 0.8) -> None:
        self.threshold = boundary_threshold
        self.max_silence = max_silence_sec
        self.lbdm = LBDM()
        
        self.current_phrase: List[int] = []
        self.last_note_time: float = time.time()
        self.dictionary: Dict[int, List[List[int]]] = {}
        self.phrase_counter: int = 0

    def add_note(self, midi_note: int) -> bool:
        """
        Add a note to the current stream and check if a phrase boundary occurred.
        
        Args:
            midi_note: Incoming MIDI note number (0 represents a flush/silence trigger).
            
        Returns:
            True if a phrase boundary was detected and dictionary was updated, False otherwise.
        """
        now = time.time()
        time_since_last = now - self.last_note_time
        self.last_note_time = now

        # If it is a dummy flush note (0) or silence trigger
        if midi_note == 0:
            if len(self.current_phrase) > 0:
                self._save_phrase()
                return True
            return False

        self.current_phrase.append(midi_note)

        # Gestalt Rule 1: Long silence indicates phrase completion
        if time_since_last > self.max_silence and len(self.current_phrase) > 1:
            # Save all notes except the one just played (which starts the new phrase)
            new_note = self.current_phrase.pop()
            self._save_phrase()
            self.current_phrase.append(new_note)
            return True

        # Gestalt Rule 2: Large interval jump (simple real-time heuristic approximation)
        if len(self.current_phrase) >= 3:
            jump = abs(self.current_phrase[-1] - self.current_phrase[-2])
            if jump >= 12:  # Octave leap or greater
                new_note = self.current_phrase.pop()
                self._save_phrase()
                self.current_phrase.append(new_note)
                return True

        return False

    def _save_phrase(self) -> None:
        """Save the completed melodic phrase into the internal dictionary."""
        if not self.current_phrase:
            return
            
        self.phrase_counter += 1
        length = len(self.current_phrase)
        
        if length not in self.dictionary:
            self.dictionary[length] = []
            
        self.dictionary[length].append(list(self.current_phrase))
        self.current_phrase.clear()