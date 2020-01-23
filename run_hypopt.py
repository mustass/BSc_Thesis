import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *
from core.create_folder import *
from core.pred_sequence import *
import matplotlib.pyplot as plt
import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

# detect the current working directory and print it
path = os.path.dirname(os.path.abspath(__file__))
print("The current working directory is %s" % path)

ray.init()
track.init()
space = {
    "lr": hp.loguniform('lr', np.log(0.0001), np.log(1)),
    "timesteps": hp.choice('timesteps', range(10, 50, 1)),
    "num_layers": hp.choice('num_layers', range(1, 5, 1)),
    "hidden_dim": hp.choice('hidden_dim', range(1, 5, 1)),
}


algo = HyperOptSearch(
    space, max_concurrent=4, metric="error", mode="max")


sched = AsyncHyperBandScheduler(
    metric='error', mode='max', grace_period=10)
config = {
    "name": "Jan21",
    "stop": {
        "error": -0.00001,
        "training_iteration": 150
    },
    "config": {"filename": '/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/DJI.csv',
               "path": path, }
}

analysis = tune.run(
    train_hypopt,
    search_alg=algo,
    num_samples= 50,
    scheduler=sched,
    resources_per_trial={
        "cpu": 8,
        "gpu": 1
    },
    **config)

print("Best config is:", analysis.get_best_config(metric="error"))
