# speed_test.py
import time
import torch
from backend.taxi.state_utils import describe_state_for_llm, get_prompt
from backend.llm.client import get_qwen_action, _get_client, _ensure_pipeline_ready

def wait_for_model_loading():
    """Wait for model to fully load"""
    print("🔄 Waiting for model to load...")
    client = _get_client()
    
    # Force load model
    if client._pipeline is None:
        print("Forcing model load...")
        client._load_pipeline()
    
    # Wait for model to be ready
    max_wait = 180  # 3 minutes timeout
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if _ensure_pipeline_ready(client) and client._pipeline is not None:
            print("✓ Model loaded successfully!")
            # Verify model device
            try:
                device = next(client._pipeline.model.parameters()).device
                print(f"Model running on: {device}")
                return True
            except:
                print("✓ Model loaded (device detection skipped)")
                return True
        print(f"⏳ Model loading... waited {int(time.time() - start_time)} seconds")
        time.sleep(5)
    
    print("❌ Model loading timeout")
    return False

def improved_speed_test():
    print("🚕 Taxi-v3 Qwen3-0.6B Speed Test (Colab GPU)")
    print("=" * 60)
    
    # Check GPU availability
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")
    
    # Wait for model to load
    if not wait_for_model_loading():
        return
    
    # Test configuration
    EXAMPLE_DECODED_STATE = {
        "taxi_row": 2,
        "taxi_col": 1, 
        "passenger_location": 4,  # Passenger in taxi
        "destination_index": 3,   # Destination: Blue (4,3)
    }
    
    # Build prompt
    state_desc = describe_state_for_llm(EXAMPLE_DECODED_STATE)
    prompt = get_prompt(state_desc)
    
    print(f"\n📝 Test Configuration:")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"State description: {state_desc}")
    print(f"Prompt preview: {prompt[:150]}...")
    
    # Warm-up phase
    print(f"\n🔥 Warm-up phase (3 calls)...")
    warmup_times = []
    for i in range(3):
        start = time.perf_counter()
        result = get_qwen_action(prompt)
        elapsed = time.perf_counter() - start
        warmup_times.append(elapsed)
        
        action = result.get("action")
        reasoning = result.get("thinking_process", "None")
        reasoning_preview = reasoning[:80] + "..." if len(reasoning) > 80 else reasoning
        
        print(f"Warm-up {i+1}: action={action}, time={elapsed:.2f}s")
        print(f"      Reasoning: {reasoning_preview}")
    
    # Main test phase
    print(f"\n📊 Main test (5 calls)...")
    test_times = []
    actions = []
    
    for i in range(5):
        start = time.perf_counter()
        result = get_qwen_action(prompt)
        elapsed = time.perf_counter() - start
        test_times.append(elapsed)
        
        action = result.get("action")
        actions.append(action)
        reasoning = result.get("thinking_process", "None")
        reasoning_preview = reasoning[:60] + "..." if len(reasoning) > 60 else reasoning
        
        print(f"Test {i+1}: action={action}, time={elapsed:.2f}s")
        print(f"      Reasoning: {reasoning_preview}")
    
    # Results analysis
    print(f"\n📈 Detailed Results Analysis")
    print("=" * 40)
    
    if test_times:
        avg_time = sum(test_times) / len(test_times)
        min_time = min(test_times)
        max_time = max(test_times)
        std_time = (sum((t - avg_time) ** 2 for t in test_times) / len(test_times)) ** 0.5
        
        print(f"Test count: {len(test_times)}")
        print(f"Average inference time: {avg_time:.2f}s")
        print(f"Minimum time: {min_time:.2f}s")
        print(f"Maximum time: {max_time:.2f}s") 
        print(f"Time standard deviation: {std_time:.2f}s")
        
        # Action analysis
        action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
        valid_actions = [a for a in actions if a is not None and 0 <= a <= 5]
        valid_percentage = len(valid_actions) / len(actions) * 100
        
        print(f"\n🎯 Action Validity:")
        print(f"Valid actions: {len(valid_actions)}/{len(actions)} ({valid_percentage:.1f}%)")
        
        if valid_actions:
            action_counts = {}
            for action in valid_actions:
                action_name = action_names[action]
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
            
            print("Action distribution:")
            for action_name, count in action_counts.items():
                percentage = count / len(valid_actions) * 100
                print(f"  {action_name}: {count} times ({percentage:.1f}%)")
        
        # Performance comparison
        if avg_time > 0:
            speedup = 461.79 / avg_time
            print(f"\n⚡ Performance Comparison:")
            print(f"Compared to local Mac: {speedup:.1f}x faster")
            
            # Comparison with pre-optimization
            original_avg = 58.18  # Pre-optimization average time
            improvement = original_avg / avg_time
            print(f"Compared to pre-optimization: {improvement:.1f}x faster")
    
    else:
        print("❌ No valid test data")
    
    print(f"\n💡 Test completed! Project code was not modified.")

if __name__ == "__main__":
    improved_speed_test()