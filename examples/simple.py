import numpy as np
import physbo

# Make a set of candidates, test_X
D = 3  # The number of params (the dimension of parameter space)
N = 1000  # The number of candidates
test_X = np.random.randn(N, D)  # Generated from Gaussian
test_X[0, :] = 0.0  # true solution


def simulator(actions: np.ndarray) -> np.ndarray:
    """ Objective function

    Quadratic function, -Σ_i x_i^2
    Receives an array of actions (indices of candidates) and returns the corresponding results as an array
    """
    return -np.sum(test_X[actions, :] ** 2, axis=1)


policy = physbo.search.discrete.policy(test_X)
policy.set_seed(12345)

# Random search (10 times)
policy.random_search(max_num_probes=10, simulator=simulator)

# Bayesian search (40 times)
#   score function (acquition function): expectation of improvement (EI)
policy.bayes_search(max_num_probes=40, simulator=simulator, score="EI")

# Print the best result
# best_actions[i] and best_fx[i] stores the best action and value up to the i-th search (random + bayes)
best_fx, best_actions = policy.history.export_sequence_best_fx()
print(f"best_fx: {best_fx[-1]} at {test_X[best_actions[-1], :]}")
