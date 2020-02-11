import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *
from core.create_folder import *
from core.pred_sequence import *
import matplotlib.pyplot as plt
from datetime import date
import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

# detect the current working directory and print it
path = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results'

for num_forward in range(14, 15, 1):
    today = date.today()

    ray.init(ignore_reinit_error=True)
    track.init()
    space = {
        "lr": hp.loguniform('lr', np.log(0.0001), np.log(0.3)),
        "timesteps": hp.choice('timesteps', range(5, 50, 1)),
        "num_layers": hp.choice('num_layers', range(1, 5, 1)),
        "hidden_dim": hp.choice('hidden_dim', range(1, 5, 1)),
    }

    algo = HyperOptSearch(
        space, max_concurrent=4, metric="error", mode="max")

    sched = AsyncHyperBandScheduler(
        metric='error', mode='max', grace_period=10)
    config = {
        "name": str(num_forward) + "forward",
        "stop": {
            "error": -0.00001,
            "training_iteration": 100
        },
        "config": {"filename": '/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/DJI.csv',
                   "path": path,
                   "window_normalisation": True,
                   "num_forward": num_forward, }
    }

    analysis = tune.run(
        train_hypopt,
        resume=True,
        search_alg=algo,
        num_samples=50,
        scheduler=sched,
        resources_per_trial={
            "cpu": 8,
            "gpu": 1
        },
        **config)
    print("Best config is:", analysis.get_best_config(metric="error"))