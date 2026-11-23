import gymnasium as gym

# Create the MountainCar environment
def create_environment(render=False):
    if render:
        env = gym.make("MountainCar-v0", render_mode="human")
    else:
        env = gym.make("MountainCar-v0")
    return env

# Reset car to its starting position
def reset_environment(env):
    state, info = env.reset()
    return state

# Move the car using an action
def take_action(env, action):
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return state, reward, done

# Close the environment
def close_environment(env):
    env.close()
