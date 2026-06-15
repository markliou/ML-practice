import sys
import os

# Ensure local directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from space_invaders import JaxAtariEnv
import jax

def test_environment():
    print("=== Starting Atari Space Invaders Environment Test ===")
    
    try:
        env = JaxAtariEnv('SpaceInvadersNoFrameskip-v4')
        print("Successfully initialized JaxAtariEnv.")
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        print("Ensure 'gymnasium[atari,accept-rom-license]' is installed in the container.")
        sys.exit(1)

    # 1. Action Space Test
    print("\n[Action Space]")
    print(f"Action Space: {env.action_space}")
    if hasattr(env.action_space, 'n'):
        print(f"Number of actions: {env.action_space.n}")
        if hasattr(env.env, 'unwrapped') and hasattr(env.env.unwrapped, 'get_action_meanings'):
            print(f"Action meanings: {env.env.unwrapped.get_action_meanings()}")

    # 2. Observation Space Test
    print("\n[Observation Space]")
    print(f"Observation Space Shape: {env.observation_space.shape}")
    print(f"Observation Space DataType: {env.observation_space.dtype}")

    # 3. Reset Test
    print("\n[Resetting Environment]")
    obs, info = env.reset()
    print(f"Observation Type: {type(obs)}")
    print(f"Observation Shape: {obs.shape}")
    print(f"Observation Device: {obs.device}")
    print(f"Initial Info: {info}")

    # 4. Step Test
    print("\n[Simulating 5 steps]")
    for step in range(1, 6):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        
        print(f"Step {step}:")
        print(f"  Action: {action} (Meaning: {env.env.unwrapped.get_action_meanings()[action] if hasattr(env.env.unwrapped, 'get_action_meanings') else 'N/A'})")
        print(f"  Reward: {reward} (Type: {type(reward)}, Device: {reward.device})")
        print(f"  Done: {done} (Type: {type(done)}, Device: {done.device})")
        print(f"  Lives: {info.get('lives', 'N/A')}")
        
        if done:
            print("  Episode finished, resetting.")
            env.reset()
            
    env.close()
    print("\n=== Atari Test Completed Successfully ===")

if __name__ == "__main__":
    test_environment()
