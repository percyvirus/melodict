"""
Basic-Pitch Backend Benchmark
Compares the inference latency between ONNX and CoreML runtimes under sustained loads.
"""
import argparse
import os
import tempfile
import time

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Benchmark basic-pitch backends.")
    parser.add_argument("--engine", choices=["onnx", "coreml"], required=True, 
                        help="Choose the runtime engine to benchmark")
    parser.add_argument("--duration", type=float, default=30.0, 
                        help="Duration of dummy audio in seconds")
    parser.add_argument("--iterations", type=int, default=10, 
                        help="Number of inference iterations")
    args = parser.parse_args()

    # Set environment variables BEFORE importing basic-pitch
    os.environ["BASIC_PITCH_TF"] = "0"
    if args.engine == "onnx":
        os.environ["BASIC_PITCH_ONNX"] = "1"
        os.environ["BASIC_PITCH_COREML"] = "0"
    else:
        os.environ["BASIC_PITCH_ONNX"] = "0"
        os.environ["BASIC_PITCH_COREML"] = "1"

    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
    from basic_pitch.inference import predict

    # Generate dummy white noise at 44100 Hz
    sr = 44100
    dummy_audio = np.random.uniform(-1, 1, int(sr * args.duration)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
        sf.write(temp_wav.name, dummy_audio, sr)

        print(f"\n[INIT] Warming up {args.engine.upper()} engine with {args.duration}s of audio...")
        _ = predict(temp_wav.name)

        print(f"[TEST] Running {args.iterations} consecutive inferences...")
        
        start_time = time.time()
        for _ in range(args.iterations):
            _ = predict(temp_wav.name)
        end_time = time.time()

    avg_time = (end_time - start_time) / args.iterations
    print("-" * 40)
    print(f"Engine: {args.engine.upper()}")
    print(f"Audio Duration: {args.duration}s")
    print(f"Average inference latency: {avg_time * 1000:.2f} ms")
    print("-" * 40)

if __name__ == "__main__":
    main()