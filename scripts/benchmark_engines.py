"""Automated benchmarking suite to compare accuracy and latency across SOTA engines."""

import argparse
import time
import numpy as np
from melodict.extraction.sota_models import EngineFactory


def generate_test_tone(freq: float, duration_sec: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate a clean sine wave with light white noise for realism."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, 0.02, tone.shape)
    return (tone + noise).astype(np.float32)


def run_benchmark(buffer_size_ms: int = 46) -> None:
    """Run a latency and pitch detection benchmark across all available engines."""
    sample_rate = 44100
    buffer_samples = int(sample_rate * (buffer_size_ms / 1000.0))
    
    print(f"\n--- Melodict Engine Benchmark (Buffer: {buffer_size_ms}ms / {buffer_samples} samples) ---")
    
    # Target: A4 (440 Hz -> MIDI Note 69)
    target_freq = 440.0
    expected_midi = 69
    test_buffer = generate_test_tone(target_freq, duration_sec=(buffer_size_ms / 1000.0), sample_rate=sample_rate)
    
    engines = EngineFactory.get_all_available()
    if not engines:
        print("No engines could be loaded. Check dependencies.")
        return

    print(f"\nTesting against Target: {target_freq}Hz (Expected MIDI: {expected_midi})\n")
    print(f"{'Engine Name':<18} | {'Predicted MIDI':<15} | {'Latency (ms)':<15} | {'Status':<10}")
    print("-" * 65)

    for engine in engines:
        start_time = time.perf_counter()
        # Run 5 warmup cycles, then measure
        for _ in range(5):
            _ = engine.predict_frame(test_buffer, sample_rate)
            
        start_time = time.perf_counter()
        predicted = engine.predict_frame(test_buffer, sample_rate)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        status = "PASS" if predicted == expected_midi else "FAIL"
        pred_str = str(predicted) if predicted is not None else "None"
        
        print(f"{engine.name:<18} | {pred_str:<15} | {latency_ms:>10.2f} ms | {status:<10}")
    print("-" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Melodict pitch tracking engines.")
    parser.add_argument("--buffer_ms", type=int, default=46, help="Buffer size in milliseconds (default: ~2048 samples at 44.1kHz).")
    args = parser.parse_args()
    
    run_benchmark(args.buffer_ms)
