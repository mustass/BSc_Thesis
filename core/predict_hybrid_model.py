"""
This file is used to predict using the hybrid model.
The hybrid model is constructed of *n* models, to predict *n* steps forward.
The *n* models are each trained to predict:
1: 1 forward
2: 2 forward
.....
n: n forward

Arguments:
-- The 20 models
-- Dataset

"""

