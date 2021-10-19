# Environment
Easiest using `conda`:
```
conda create --name ppo --file spec-file.txt
conda activate ppo
python main.py <hyperparameters>
```
# Experiment tracking
To see different runs do `mlflow ui` and open your browser at [http://localhost:5000](http://localhost:5000).