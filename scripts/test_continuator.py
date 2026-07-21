"""Simulate an interactive call-and-response jazz session using the VMM Continuator."""

import time
from melodict.generation.continuator import VMMContinuator


def run_call_and_response_demo() -> None:
    """Simulate feeding jazz vocabulary to the Continuator and testing its reactive responses."""
    print("\n--- Melodict VMM Continuator Call-and-Response Test ---")
    
    # 1. Initialize the Continuator (looking back up to 4 notes for context)
    continuator = VMMContinuator(max_order=4)
    
    # 2. Feed a mini-corpus of standard jazz vocabulary (e.g., ii-V-I licks in C major / A minor)
    jazz_corpus = [
        [62, 65, 67, 69, 67, 64, 60],  # Dm7 -> G7 -> Cmaj7 phrasing
        [60, 62, 64, 67, 69, 72, 71],  # Ascending pentatonic run
        [72, 71, 69, 67, 65, 64, 62],  # Descending bebop scale fragment
        [67, 69, 67, 65, 64, 62, 60],  # Bluesy turnaround
        [62, 65, 67, 69, 72, 74, 76],  # High register extension
    ]
    
    print("Training VMM Continuator on foundational jazz phrasing corpus...")
    for phrase in jazz_corpus:
        continuator.learn_phrase(phrase)
    print(f"Training complete. Vocabulary size: {len(continuator.vocabulary)} notes.\n")
    print("-" * 65)
    
    # 3. Simulate real-time prompts played by the musician
    test_prompts = [
        ("Musician plays ascending opening", [60, 62, 64, 67]),
        ("Musician plays turnaround lick", [69, 67, 65, 64]),
        ("Musician plays short high call", [72, 74]),
    ]
    
    for label, prompt in test_prompts:
        print(f"\nPROMPT: {label}")
        print(f"  -> Input MIDI Sequence: {prompt}")
        
        start_time = time.perf_counter()
        # Generate an answer matching the length of the prompt
        answer = continuator.generate_continuation(input_phrase=prompt, target_length=5, temperature=0.6)
        calc_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        print(f"  <- AI Answer Sequence : {answer}")
        print(f"     [Generation Latency: {calc_time_ms:.3f} ms | Backoff Context Applied]")
        
        # Learn from the musician's prompt on-the-fly to expand the vocabulary
        continuator.learn_phrase(prompt)
        
    print("\n" + "-" * 65 + "\n")


if __name__ == "__main__":
    run_call_and_response_demo()