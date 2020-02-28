import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.evaluating import *
from plots.plots import *
from core.create_folder import *
from core.pred_sequence import *
import matplotlib.pyplot as plt
from core.training import *
from datetime import date
import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

# detect the current working directory and print it
path = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results'

for num_forward in range(1, 2, 1):
    today = date.today()

    ray.init(ignore_reinit_error=True)
    track.init()
    space = {
        "lr": hp.loguniform('lr', np.log(0.0001), np.log(0.5)),
        "timesteps": hp.choice('timesteps', range(5, 40, 1)),
        "num_layers": hp.choice('num_layers', range(1, 10, 1)),
        "hidden_dim": hp.choice('hidden_dim', range(1, 10, 1)),
    }

    algo = HyperOptSearch(
        space, max_concurrent=4, metric="error", mode="max"
    )

    sched = AsyncHyperBandScheduler(
        metric='error', mode='max', grace_period=20)
    config = {
        "name": str(num_forward) + "forward_returns_N225",
        "stop": {
            "error": -0.00001,
            "training_iteration": 100
        },
        "config": {"filename": '/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/N225.csv',
                   "path": path,
                   "window_normalisation": False,
                   "num_forward": num_forward, }
    }

    analysis = tune.run(
        train_hypopt,
        resume=False,
        search_alg=algo,
        num_samples=10,
        scheduler=sched,
        resources_per_trial={
            "cpu": 8,
            "gpu": 1
        },
        **config)
    print("Best config is:", analysis.get_best_config(metric="error"))
