"""Modular SOTA pitch tracking engines for multi-algorithm benchmarking and real-time inference."""

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout
from typing import ClassVar

import numpy as np
import soundfile as sf

logger = logging.getLogger("MelodictEngines")


class PitchExtractorEngine(ABC):
    """Abstract base class for all pitch extraction and melody tracking engines."""

    def __init__(self, name: str, min_freq: float = 60.0, max_freq: float = 2000.0, context_sec: float = 0.0) -> None:
        self.name = name
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.context_sec = context_sec
        self.is_ready = False
        self._history_buffer = np.array([], dtype=np.float32)

    def _get_context_buffer(self, new_frame: np.ndarray, sample_rate: int) -> np.ndarray:
        """Maintains a sliding window buffer to provide historical context to ML models."""
        if self.context_sec <= 0:
            return new_frame
            
        max_samples = int(self.context_sec * sample_rate)
        
        if len(self._history_buffer) == 0:
            self._history_buffer = new_frame
        else:
            self._history_buffer = np.concatenate((self._history_buffer, new_frame))
            
        if len(self._history_buffer) > max_samples:
            self._history_buffer = self._history_buffer[-max_samples:]
            
        return self._history_buffer

    def reset_context(self) -> None:
        """Clears the history buffer. Essential when switching to a new audio file or song."""
        self._history_buffer = np.array([], dtype=np.float32)

    @abstractmethod
    def load_model(self) -> bool:
        """Load underlying neural networks or acoustic algorithms into memory."""

    @abstractmethod
    def predict_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> int | None:
        """Predict the dominant symbolic MIDI note from an audio buffer frame."""

    def freq_to_midi(self, freq: float) -> int | None:
        """Utility to convert frequency in Hz to nearest symbolic MIDI integer."""
        if freq < self.min_freq or freq > self.max_freq or np.isnan(freq) or freq <= 0:
            return None
        midi_float = 69.0 + 12.0 * np.log2(freq / 440.0)
        return int(np.round(midi_float))


class LibrosaPyinEngine(PitchExtractorEngine):
    def __init__(self) -> None:
        super().__init__(name="librosa-pyin", context_sec=0.0)

    def load_model(self) -> bool:
        logger.info("Initializing Librosa pYIN acoustic engine...")
        self.is_ready = True
        return True

    def predict_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> int | None:
        if not self.is_ready or len(audio_buffer) == 0:
            return None
        try:
            import librosa
            
            f0, _, _ = librosa.pyin(
                audio_buffer,
                fmin=self.min_freq,
                fmax=self.max_freq,
                sr=sample_rate,
                frame_length=len(audio_buffer),
            )
            valid_f0 = [f for f in f0 if f is not None and not np.isnan(f) and f > 0]
            if not valid_f0:
                return None
            return self.freq_to_midi(float(np.median(valid_f0)))
        except ImportError as e:
            logger.error(f"Missing dependency for pYIN: {e}")
            return None


class TorchCrepeEngine(PitchExtractorEngine):
    def __init__(self, model_capacity: str = "tiny") -> None:
        super().__init__(name=f"crepe-{model_capacity}", context_sec=0.0)
        self.capacity = model_capacity
        self.device = "cpu"

    def load_model(self) -> bool:
        logger.info(f"Loading TorchCrepe ({self.capacity}) weights...")
        try:
            import torch
            
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.is_ready = True
            return True
        except ImportError as e:
            logger.warning(f"torchcrepe failed to load: {e}. Engine disabled.")
            return False

    def predict_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> int | None:
        if not self.is_ready or len(audio_buffer) == 0:
            return None
        try:
            import torch
            import torchcrepe
            
            audio_tensor = torch.tensor(audio_buffer, dtype=torch.float32, device=self.device).unsqueeze(0)
            if sample_rate != 16000:
                import torchaudio.transforms as T
                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000).to(self.device)
                audio_tensor = resampler(audio_tensor)
                
            pitch, periodicity = torchcrepe.predict(
                audio_tensor,
                sample_rate=16000,
                hop_length=len(audio_buffer),
                fmin=self.min_freq,
                fmax=self.max_freq,
                model=self.capacity,
                batch_size=1,
                device=self.device,
                return_periodicity=True,
            )
            
            if periodicity.max().item() < 0.5:
                return None
            return self.freq_to_midi(float(pitch.median().item()))
        except Exception as e:  # noqa: BLE001
            logger.error(f"CREPE prediction error: {e}")
            return None


class BasicPitchEngine(PitchExtractorEngine):
    def __init__(self) -> None:
        super().__init__(name="basic-pitch", context_sec=2.0)

    def load_model(self) -> bool:
        logger.info("Loading Spotify Basic-Pitch polyphonic model (ONNX)...")
        try:
            import os
            os.environ["BASIC_PITCH_TF"] = "0"
            os.environ["BASIC_PITCH_ONNX"] = "1"
            
            root_logger = logging.getLogger()
            previous_level = root_logger.getEffectiveLevel()
            root_logger.setLevel(logging.ERROR)
            
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
            
            root_logger.setLevel(previous_level)
            
            self._predict_fn = predict
            self._model_path = ICASSP_2022_MODEL_PATH
            self.is_ready = True
            return True
        except ImportError as e:
            logger.warning(f"basic-pitch failed to load: {e}. Engine disabled.")
            return False

    def predict_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> int | None:
        if not self.is_ready or len(audio_buffer) == 0:
            return None
            
        context_buffer = self._get_context_buffer(audio_buffer, sample_rate)
        buffer_duration = len(context_buffer) / sample_rate
        new_frame_duration = len(audio_buffer) / sample_rate
        
        try:
            import os
            import tempfile
            from contextlib import redirect_stderr, redirect_stdout
            
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                sf.write(temp_wav.name, context_buffer, sample_rate)
                
                with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
                    _, midi_data, _ = self._predict_fn(temp_wav.name)
                
                if midi_data.instruments and midi_data.instruments[0].notes:
                    notes = midi_data.instruments[0].notes
                    active_notes = []
                    current_chunk_start_time = buffer_duration - new_frame_duration
                    
                    for n in notes:
                        if n.start <= buffer_duration and n.end >= current_chunk_start_time:
                            freq = 440.0 * (2.0 ** ((n.pitch - 69) / 12.0))
                            if self.min_freq <= freq <= self.max_freq:
                                active_notes.append(n)
                                
                    if active_notes:
                        best_note = max(active_notes, key=lambda n: n.pitch)
                        return best_note.pitch
                        
            return None
        except Exception as e:  # noqa: BLE001
            logger.error(f"Basic-Pitch prediction error: {e}")
            return None
    
    def predict_polyphonic_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> list[int]:
        """Predict ALL dominant symbolic MIDI notes from an audio buffer frame for polyphonic evaluation."""
        if not self.is_ready or len(audio_buffer) == 0:
            return []
            
        context_buffer = self._get_context_buffer(audio_buffer, sample_rate)
        buffer_duration = len(context_buffer) / sample_rate
        new_frame_duration = len(audio_buffer) / sample_rate
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                sf.write(temp_wav.name, context_buffer, sample_rate)
                
                with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
                    _, midi_data, _ = self._predict_fn(temp_wav.name)
                
                active_notes = []
                if midi_data.instruments and midi_data.instruments[0].notes:
                    notes = midi_data.instruments[0].notes
                    current_chunk_start_time = buffer_duration - new_frame_duration
                    
                    for n in notes:
                        # Check if the note intersects with our strict 46ms causality window
                        if n.start <= buffer_duration and n.end >= current_chunk_start_time:
                            freq = 440.0 * (2.0 ** ((n.pitch - 69) / 12.0))
                            if self.min_freq <= freq <= self.max_freq:
                                active_notes.append(n.pitch)
                                
                return list(set(active_notes))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Basic-Pitch polyphonic prediction error: {e}")
            return []


class EssentiaYinEngine(PitchExtractorEngine):
    def __init__(self) -> None:
        super().__init__(name="essentia-yin", context_sec=0.0)

    def load_model(self) -> bool:
        logger.info("Initializing Essentia PitchYin C++ engine...")
        try:
            import essentia.standard as es  # noqa: F401
            self.is_ready = True
            return True
        except ImportError:
            logger.warning("essentia not installed. Engine disabled.")
            return False

    def predict_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> int | None:
        if not self.is_ready or len(audio_buffer) == 0:
            return None
        try:
            import essentia.standard as es
            
            pitch_extractor = es.PitchYin(
                frameSize=len(audio_buffer),
                sampleRate=sample_rate,
                minFrequency=self.min_freq,
                maxFrequency=self.max_freq
            )
            pitch, confidence = pitch_extractor(audio_buffer.astype(np.float32))
            
            if confidence < 0.5 or pitch == 0 or np.isnan(pitch):
                return None
            return self.freq_to_midi(float(pitch))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Essentia YIN prediction error: {e}")
            return None


class EngineFactory:
    # Solves RUF012: Explicitly marking class-level mutable dictionary
    _ENGINES: ClassVar[dict[str, type[PitchExtractorEngine]]] = {
        "pyin": LibrosaPyinEngine,
        "crepe": TorchCrepeEngine,
        "basic-pitch": BasicPitchEngine,
        "essentia-yin": EssentiaYinEngine,
    }

    @classmethod
    def create(cls, engine_name: str) -> PitchExtractorEngine | None:
        engine_class = cls._ENGINES.get(engine_name.lower())
        if not engine_class:
            return None
        engine = engine_class()
        engine.load_model()
        return engine
        
    @classmethod
    def get_all_available(cls) -> list[PitchExtractorEngine]:
        engines: list[PitchExtractorEngine] = []
        for name in cls._ENGINES:
            engine = cls.create(name)
            if engine and engine.is_ready:
                engines.append(engine)
        return engines