from matplotlib import pyplot as plt

import gym

env = gym.envs.make("Breakout-v0")

print("Action space size: {}".format(env.action_space.n))
# print(env.get_action_meanings())
observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

#plt.figure()
#plt.imshow(env.render(mode='rgb_array'))
#[env.step(2) for x in range(1)]
#plt.figure()
#plt.imshow(env.render(mode='rgb_array'))
#env.render()
## Check out what a cropped image looks like
#plt.imshow(observation[34:-16,:,:])

env = gym.make('SpaceInvaders-v0')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
    env.render('human')
env.close()  # https://github.com/openai/gym/issues/893
