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


path = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/returns/hyperopt'

for dataset in ["DJI", "GSPC", "IXIC", "N225"]:
    today = date.today()
    ray.init(ignore_reinit_error=True)
    track.init()
    space = {
        "lr": hp.loguniform('lr', np.log(0.0001), np.log(0.5)),
        "timesteps": hp.choice('timesteps', range(5, 40, 1)),
        "num_layers": hp.choice('num_layers', range(1, 5, 1)),
        "hidden_dim": hp.choice('hidden_dim', range(1, 15, 1)),
        "dropout": hp.uniform('dropout', 0.0, 0.5),
    }
    algo = HyperOptSearch(
        space, max_concurrent=2, metric="error", mode="max"
    )
    sched = AsyncHyperBandScheduler(
        metric='error', mode='max', grace_period=15)
    config = {
        "name": "1_forward_returns_"+dataset,
        "stop": {
            "error": -0.000001,
            "training_iteration": 100
        },
        "config": {"filename": '/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/'+dataset+'.csv',
                   "path": path+'/'+dataset,
                   "window_normalisation": False,
                   "num_forward": 1, }
    }
    analysis = tune.run(
        train_hypopt,
        resume=False,
        search_alg=algo,
        num_samples=25,
        scheduler=sched,
        resources_per_trial={
            "cpu": 8,
            "gpu": 1
        },
        **config)
    print("Best config is:", analysis.get_best_config(metric="error"))
