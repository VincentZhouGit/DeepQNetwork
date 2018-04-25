import gym
from exploration import get_epsilon

game = 'BreakoutDeterministic-v4'
episodes = 1

env = gym.make(game)

for i in range(1):
    observation = env.reset()
    steps = 0
    done = False
    while not done:
        env.render()

        epsilon = get_epsilon(i)
        
        # Choose the action 
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = choose_best_action(model, state)
        
        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        steps += 1
        if done:
            print("done == True")
            print("info: {}".format(info))

    print("steps: {}".format(steps))


# 1.  take some action ai and observe a <s1,a1,s2,r1>, and add this to replay buffer

# 2. sample mini_batch from the replay uniformaly

# 3 compute yi

# 4. update the network

# 5(skip for now) copy parameters over





def q_iteration(env, model, state, iteration, memory):
    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)

    # Choose the action 
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, state)

    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    new_frame, reward, is_done, _ = env.step(action)
    memory.add(state, action, new_frame, reward, is_done)

    # Sample and fit
    batch = memory.sample_batch(32)
    fit_batch(model, batch)