"""Variable-order Markov Model (VMM) for stylistic melodic continuation."""

import logging
import random
from typing import Dict, List, Tuple

logger = logging.getLogger("MelodictVMM")

NoteTuple = Tuple[int, int, int]  # (Pitch, Duration, Velocity)

class VMMContinuator:
    """Learns multi-dimensional transitions and generates reactive continuations."""

    def __init__(self, max_order: int = 3):
        self.max_order = max_order
        # Maps state tuple/sequence strings to a list of next possible NoteTuples
        self.transitions: Dict[str, List[NoteTuple]] = {}

    def _get_state_key(self, sequence: List[NoteTuple]) -> str:
        """Serialize a sequence of tuples into a string key."""
        return "|".join([f"{p}_{d}_{v}" for p, d, v in sequence])

    def learn_from_dictionary(self, dictionary: Dict[int, List[List[NoteTuple]]]) -> None:
        """Populate the Markov transition table from captured phrase dictionaries."""
        for phrase_list in dictionary.values():
            for phrase in phrase_list:
                for order in range(1, self.max_order + 1):
                    for i in range(len(phrase) - order):
                        context = phrase[i : i + order]
                        next_note = phrase[i + order]
                        key = self._get_state_key(context)
                        
                        if key not in self.transitions:
                            self.transitions[key] = []
                        self.transitions[key].append(next_note)

    def generate_continuation(
        self, input_phrase: List[NoteTuple], target_length: int = 8, temperature: float = 0.7
    ) -> List[NoteTuple]:
        """Generate a response sequence matching the rhythm and dynamics of the input."""
        if not self.transitions:
            return input_phrase[:target_length]

        response: List[NoteTuple] = []
        # Start matching from the end of the input phrase
        current_context = input_phrase.copy()

        for _ in range(target_length):
            next_note = None
            # Backoff strategy: try matching largest order first, down to order 1
            for order in range(min(self.max_order, len(current_context)), 0, -1):
                key = self._get_state_key(current_context[-order:])
                if key in self.transitions and self.transitions[key]:
                    choices = self.transitions[key]
                    next_note = random.choice(choices)
                    break
            
            # Fallback if no context match is found in memory
            if not next_note:
                all_notes = [note for note_list in self.transitions.values() for note in note_list]
                if not all_notes: break
                next_note = random.choice(all_notes)

            response.append(next_note)
            current_context.append(next_note)

        return response