# Pybrook : A tool to skull strip MRI images
<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/pybrook_logo.jpg?raw=true" alt="example0">
</p>


**Pybrook** is a python package designed for medical MRI preprocessing. Specifically Pybrook is designed to automatically extract the brain from MRI images. This package has a lack of (modern) tools to address this problem. By using several models (Resnet, Efficient net) Pybrook achieves an IOU score of **0.98**. 

## Data and training


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
