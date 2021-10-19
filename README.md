# PPO on CarRacing-v0
This repo contains code to train a [PPO](https://arxiv.org/abs/1707.06347) agent on the [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) environment from OpenAI.

## Environment
It's easiest to set up the environment using `conda`:
```
conda create --name ppo --file spec-file.txt
conda activate ppo
```
## Training
To run the training code do
```
python main.py <hyperparameters>
```
## Tracking
To see different runs do `mlflow ui` and open your browser at [http://localhost:5000](http://localhost:5000).