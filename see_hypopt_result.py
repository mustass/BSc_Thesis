from ray.tune import Analysis
analysis = Analysis("~/ray_results/exp")

print(analysis.get_best_config(metric = "error", mode = "max"))