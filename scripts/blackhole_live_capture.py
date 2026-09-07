"""
Live audio capture using BlackHole virtual audio driver.
Connect Max8 output to BlackHole, and run this script to capture low-latency PCM frames.
"""
import argparse
import queue
import sys

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice numpy")
    sys.exit(1)

# Queue to hold audio frames for processing outside the audio thread
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called for each audio block by sounddevice."""
    if status:
        print(status, file=sys.stderr)
    
    # Mixdown to mono if stereo
    mono_data = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
    audio_queue.put(mono_data.copy())

def run_blackhole_capture(device_name="BlackHole", samplerate=44100, buffer_ms=46):
    """Listen to BlackHole and pass frames to the extraction engine."""
    blocksize = int(samplerate * (buffer_ms / 1000.0))
    
    # Check if device exists
    try:
        devices = sd.query_devices()
    except Exception as e:  # noqa: BLE001
        print(f"Error querying audio devices: {e}")
        return

    device_idx = None
    for idx, d in enumerate(devices):
        if device_name.lower() in d['name'].lower() and d['max_input_channels'] > 0:
            device_idx = idx
            break
            
    if device_idx is None:
        print(f"Error: Could not find input device matching '{device_name}'.")
        print("Available input devices:")
        print(sd.query_devices())
        return

    print(f"Starting native capture on: {devices[device_idx]['name']} (Buffer: {buffer_ms}ms)")
    print("Waiting for audio from Max8... (Press Ctrl+C to stop)")

    try:
        with sd.InputStream(device=device_idx, channels=2, samplerate=samplerate, 
                            blocksize=blocksize, callback=audio_callback):
            while True:
                # Get block from queue
                audio_frame = audio_queue.get()
                
                # --- Here is where basic-pitch/essentia predict_frame() goes ---
                # Example: checking volume to show it's working
                rms = np.sqrt(np.mean(audio_frame**2))
                if rms > 0.005:  # Noise floor threshold
                    print(f"Received audio frame (size: {len(audio_frame)}). RMS Level: {rms:.4f}")
                    
    except KeyboardInterrupt:
        print("\nStopping BlackHole capture.")
    except Exception as e:  # noqa: BLE001
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlackHole Low-Latency Audio Capture")
    parser.add_argument("--device", type=str, default="BlackHole", help="Name of the BlackHole device")
    parser.add_argument("--buffer_ms", type=int, default=46, help="Buffer size in milliseconds")
    args = parser.parse_args()
    
    run_blackhole_capture(args.device, args.buffer_ms)