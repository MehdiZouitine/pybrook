# PyBrook : A tool for skull strip MRI images
<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/pybrook_logo.jpg?raw=true" alt="example0">
</p>


## Skull stripping blend of SOTA models : 

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/example0.gif?raw=true" alt="example0">
</p>

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/example1.gif?raw=true" alt="example1">
</p>

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/example2.gif?raw=true" alt="example2">
</p>

### How to install ?
```
git clone https://github.com/MehdiZouitine/gym_ma_toy
cd gym_ma_toy
pip install -e .
```


### How to use it ?

```python
import gym
import gym_ma_toy

env = gym.make('team_catcher-v0')

obs = env.reset()
done = False
while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
env.close()
```
