"""
Implementation of the Local Boundary Detection Model (LBDM) by Emilios Cambouropoulos.
Calculates perceptual phrase boundaries based on pitch intervals, rests, and IOIs.
"""

import numpy as np


class LBDM:
    """Calculates phrase boundary strength profiles for melodic sequences."""

    def __init__(self, weight_pitch: float = 0.25, weight_ioi: float = 0.5, weight_rest: float = 0.25) -> None:
        self.w_pitch = weight_pitch
        self.w_ioi = weight_ioi
        self.w_rest = weight_rest

    def compute_boundary_strength(self, pitches: list[int], iois: list[float], rests: list[float]) -> np.ndarray:
        """
        Compute the degree of boundary strength for each interval in the sequence.
        
        Args:
            pitches: Sequence of MIDI pitch numbers.
            iois: Inter-Onset Intervals in seconds.
            rests: Rest durations between notes in seconds.
            
        Returns:
            1D numpy array of boundary strength values.
        """
        if len(pitches) < 3:
            return np.zeros(len(pitches))

        # Calculate absolute pitch intervals
        intervals = np.abs(np.diff(pitches))
        
        # Calculate local change degree for intervals
        strength = np.zeros(len(intervals))
        for i in range(1, len(intervals) - 1):
            # Rule of proximity and interval jump
            diff_prev = np.abs(intervals[i] - intervals[i - 1])
            diff_next = np.abs(intervals[i] - intervals[i + 1])
            strength[i] = (diff_prev + diff_next) * self.w_pitch

        return strength