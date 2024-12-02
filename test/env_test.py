import gymnasium as gym


# env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")

env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.3, render_mode="human")


state,  _ = env.reset()

done = False

total_reward = 0

while not done: 

    action = env.action_space.sample()  

    next_state, reward, done, _, _ = env.step(action=action)

    total_reward += reward
    
    print(total_reward)

    env.render()

env.close()
