# CoDaPO: Confidence and Difficulty-adaptive Policy Optimization.
#
# This package keeps the CoDaPO-specific training entrypoint and a copy of the
# verl Ray PPO trainer carrying the CoDaPO modifications, so that AlphaApollo's
# native trainer at alphaapollo/core/generation/verl/trainer/ppo/ray_trainer.py
# stays untouched. The CoDaPO advantage / entropy-shaping functions live in
# AlphaApollo's core_algos and are imported from there.
