import gymnasium as gym
import ale_py
import jax.numpy as jnp

class JaxAtariEnv:
    """
    A clean wrapper for Atari environments that returns JAX arrays directly.
    Designed for integration into JAX RL (e.g. PPO) training loops.
    """
    def __init__(self, env_id='SpaceInvadersNoFrameskip-v4'):
        self.env = gym.make(env_id, render_mode='rgb_array')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        """Resets the environment and returns (obs_jax, info)."""
        obs, info = self.env.reset()
        return jnp.array(obs), info

    def step(self, action):
        """
        Steps the environment.
        Accepts action (JAX array or int) and returns (obs_jax, reward_jax, done_jax, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        done = terminated or truncated
        
        return (
            jnp.array(obs), 
            jnp.array(reward, dtype=jnp.float32), 
            jnp.array(done, dtype=jnp.bool_), 
            info
        )

    def close(self):
        self.env.close()
