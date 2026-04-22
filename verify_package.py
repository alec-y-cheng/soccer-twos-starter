import sys
import numpy as np
import torch

def verify():
    print("Testing Submission package...")
    try:
        from submission.agent_ray import SubmissionAgent
        agent = SubmissionAgent()
        
        # Test with dummy observation
        dummy_obs = {
            0: np.random.rand(336).astype(np.float32),
            1: np.random.rand(336).astype(np.float32),
            2: np.random.rand(336).astype(np.float32),
            3: np.random.rand(336).astype(np.float32),
        }
        
        actions = agent.act(dummy_obs)
        print("✓ Successfully calculated actions for all PIDs.")
        print("Actions sample (PID 0):", actions[0])
        print("Actions sample (PID 1):", actions[1])
        
        # Check if Ray is in sys.modules
        ray_loaded = 'ray' in sys.modules
        print(f"Ray library loaded? {'YES (Warning)' if ray_loaded else 'NO (Success!)'}")
        
        print("✓ Submission package is ready!")
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
