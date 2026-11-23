import torch
from dqn import DQN
from environment import (
    create_environment,
    reset_environment,
    take_action,
    close_environment
)

#Create the Mountaincar environment
env = create_environment(render=True)

#Get the number of states and actions
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

#Create the DQN neural network
model = DQN(state_size, action_size)

#Reset car
state = reset_environment(env)

#Run 200 random action
for step in range(200):

    #Convert state to pyTorch tensor
    state_tensor = torch.tensor(state, dtype=torch.float32)

    #Get Q-values from nn
    with torch.no_grad():
        q_values = model(state_tensor)

    #Choose the action with highest Q-value
    action = torch.argmax(q_values).item()

    #Move the car
    state, reward, done = take_action(env, action)

    #Display information
    print(f"Step: {step + 1}")
    print(f"Position: {state[0]:.4f}")
    print(f"Velocity: {state[1]:.4f}")
    print(f"Reward: {reward}")
    print(f"Action: {action}")
    print("-----------------------------")

    if done:
        break

#Close the environment
close_environment(env)