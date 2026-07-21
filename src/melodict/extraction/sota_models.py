"""Modular SOTA pitch tracking engines for multi-algorithm benchmarking and real-time inference."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
import numpy as np

logger = logging.getLogger("MelodictEngines")


class PitchExtractorEngine(ABC):
    """Abstract base class for all pitch extraction and melody tracking engines."""

    def __init__(self, name: str, min_freq: float = 60.0, max_freq: float = 2000.0) -> None:
        self.name = name
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.is_ready = False

    @abstractmethod
    def load_model(self) -> bool:
        """Load underlying neural networks or acoustic algorithms into memory."""
        pass

    @abstractmethod
    def predict_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> Optional[int]:
        """
        Predict the dominant symbolic MIDI note from an audio buffer frame.
        
        Args:
            audio_buffer: 1D numpy array of PCM audio samples.
            sample_rate: Sampling rate of the incoming audio.
            
        Returns:
            Estimated MIDI note number or None if silence/unvoiced.
        """
        pass

    def freq_to_midi(self, freq: float) -> Optional[int]:
        """Utility to convert frequency in Hz to nearest symbolic MIDI integer."""
        if freq < self.min_freq or freq > self.max_freq or np.isnan(freq) or freq == 0:
            return None
        midi_float = 69.0 + 12.0 * np.log2(freq / 440.0)
        return int(np.round(midi_float))


class LibrosaPyinEngine(PitchExtractorEngine):
    """Acoustic probabilistic YIN engine. Best for monophonic low-latency tracking (Guitar/Solo)."""

    def __init__(self) -> None:
        super().__init__(name="librosa-pyin")

    def load_model(self) -> bool:
        logger.info("Initializing Librosa pYIN acoustic engine (no GPU weights needed)...")
        self.is_ready = True
        return True

    def predict_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> Optional[int]:
        if not self.is_ready or len(audio_buffer) == 0:
            return None
        try:
            import librosa
            # Estimate fundamental frequency using probabilistic YIN
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
        except Exception as e:
            logger.error(f"pYIN prediction error: {e}")
            return None


class TorchCrepeEngine(PitchExtractorEngine):
    """Deep learning monophonic pitch tracker using PyTorch (Metal/GPU accelerated)."""

    def __init__(self, model_capacity: str = "tiny") -> None:
        super().__init__(name=f"crepe-{model_capacity}")
        self.capacity = model_capacity
        self.device = "cpu"

    def load_model(self) -> bool:
        logger.info(f"Loading TorchCrepe ({self.capacity}) model weights...")
        try:
            # SAFETY INJECTION: Force loading pkg_resources for legacy dependencies
            import setuptools
            import pkg_resources
            import torch
            import resampy
            import torchcrepe
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"TorchCrepe initialized successfully on device: {self.device.upper()}")
            self.is_ready = True
            return True
        except Exception as e:
            logger.warning(f"torchcrepe failed to load: {e}. Engine disabled.")
            return False

    def predict_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> Optional[int]:
        if not self.is_ready or len(audio_buffer) == 0:
            return None
        try:
            import torch
            import torchcrepe
            
            # Ensure proper tensor dimensions (1, N) and sampling rate (16kHz for CREPE)
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
            return self.freq_to_midi(pitch.median().item())
        except Exception as e:
            logger.error(f"CREPE prediction error: {e}")
            return None


class BasicPitchEngine(PitchExtractorEngine):
    """Spotify's lightweight polyphonic Audio-to-MIDI model. Best for Piano."""

    def __init__(self) -> None:
        super().__init__(name="basic-pitch")

    def load_model(self) -> bool:
        logger.info("Loading Spotify Basic-Pitch polyphonic model (ONNX)...")
        try:
            # SAFETY INJECTION: Prevent mir_eval import failures and force ONNX backend
            import setuptools
            import pkg_resources
            import os
            
            # Force the Spotify engine to bypass TensorFlow and use ONNX on Apple Silicon
            os.environ["BASIC_PITCH_TF"] = "0"
            os.environ["BASIC_PITCH_ONNX"] = "1"
            
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
            self._predict_fn = predict
            self._model_path = ICASSP_2022_MODEL_PATH
            self.is_ready = True
            logger.info("Basic-Pitch engine loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"basic-pitch failed to load: {e}. Engine disabled.")
            return False

    def predict_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> Optional[int]:
        if not self.is_ready or len(audio_buffer) == 0:
            return None
        try:
            import tempfile
            import os
            import soundfile as sf
            from contextlib import redirect_stdout, redirect_stderr
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                sf.write(temp_wav.name, audio_buffer, sample_rate)
                
                # Silence basic-pitch native stdout/stderr prints during benchmarking
                with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull), redirect_stderr(fnull):
                        _, midi_data, _ = self._predict_fn(temp_wav.name)
                
                if midi_data.instruments and midi_data.instruments[0].notes:
                    notes = midi_data.instruments[0].notes
                    valid_notes = [n for n in notes if self.min_freq <= (440.0 * (2.0 ** ((n.pitch - 69) / 12.0))) <= self.max_freq]
                    if valid_notes:
                        best_note = max(valid_notes, key=lambda n: n.pitch)
                        return best_note.pitch
            return None
        except Exception as e:
            logger.error(f"Basic-Pitch prediction error: {e}")
            return None


class EssentiaYinEngine(PitchExtractorEngine):
    """MTG-UPF's C++ optimized acoustic YIN engine. Ultra-fast CPU tracking for real-time systems."""

    def __init__(self) -> None:
        super().__init__(name="essentia-yin")

    def load_model(self) -> bool:
        logger.info("Initializing Essentia PitchYin C++ engine (MTG-UPF Barcelona)...")
        try:
            import essentia.standard as es
            self.is_ready = True
            logger.info("Essentia YIN engine loaded successfully.")
            return True
        except ImportError:
            logger.warning("essentia not installed. Engine disabled.")
            return False

    def predict_frame(self, audio_buffer: np.ndarray, sample_rate: int = 44100) -> Optional[int]:
        if not self.is_ready or len(audio_buffer) == 0:
            return None
        try:
            import essentia.standard as es
            
            # Essentia executes its native C++ algorithm requiring a float32 array
            pitch_extractor = es.PitchYin(
                frameSize=len(audio_buffer),
                sampleRate=sample_rate,
                minFrequency=self.min_freq,
                maxFrequency=self.max_freq
            )
            pitch, confidence = pitch_extractor(audio_buffer.astype(np.float32))
            
            # Standard confidence threshold for real-time systems
            if confidence < 0.5 or pitch == 0 or np.isnan(pitch):
                return None
            return self.freq_to_midi(float(pitch))
        except Exception as e:
            logger.error(f"Essentia YIN prediction error: {e}")
            return None


class EngineFactory:
    """Factory to instantiate and retrieve registered pitch tracking engines."""
    
    _ENGINES: Dict[str, Type[PitchExtractorEngine]] = {
        "pyin": LibrosaPyinEngine,
        "crepe": TorchCrepeEngine,
        "basic-pitch": BasicPitchEngine,
        "essentia-yin": EssentiaYinEngine,
    }

    @classmethod
    def create(cls, engine_name: str) -> Optional[PitchExtractorEngine]:
        """Create and load the specified engine by name."""
        engine_class = cls._ENGINES.get(engine_name.lower())
        if not engine_class:
            logger.error(f"Unknown engine: {engine_name}. Available: {list(cls._ENGINES.keys())}")
            return None
        engine = engine_class()
        engine.load_model()
        return engine
        
    @classmethod
    def get_all_available(cls) -> List[PitchExtractorEngine]:
        """Instantiate and return all engines that successfully load on the system."""
        engines = []
        for name in cls._ENGINES:
            engine = cls.create(name)
            if engine and engine.is_ready:
                engines.append(engine)
        return engines