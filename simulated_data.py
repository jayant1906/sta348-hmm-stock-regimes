import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import permutations
from pathlib import Path

from plot_results import create_plots


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# True hidden states in the simulated market:
# 0 = Bull market, 1 = Bear market, 2 = Sideways market.
STATE_NAMES = {0: "Bull", 1: "Bear", 2: "Sideways"}
NUM_STATES = len(STATE_NAMES)


# Transition matrix for Bull, Bear, and Sideways states.
TRANSITION_MATRIX = np.array([
    [0.95, 0.02, 0.03],
    [0.04, 0.92, 0.04],
    [0.03, 0.03, 0.94],
])

# Initial state probabilities.
INITIAL_PROBS = np.array([1 / 3, 1 / 3, 1 / 3])

# Number of observations.
N_OBSERVATIONS = 2000
N_INITIALIZATIONS = 10

# Gaussian emission parameters.
EMISSION_MEANS = np.array([0.001, -0.0015, 0.0])
EMISSION_STD_DEVS = np.array([0.01, 0.02, 0.005])


# Simulate the hidden state sequence.
true_states = np.empty(N_OBSERVATIONS, dtype=int)
true_states[0] = np.random.choice(NUM_STATES, p=INITIAL_PROBS)

for t in range(1, N_OBSERVATIONS):
    true_states[t] = np.random.choice(
        NUM_STATES,
        p=TRANSITION_MATRIX[true_states[t - 1]],
    )


# Simulate one return for each hidden state.
simulated_returns = np.random.normal(
    loc=EMISSION_MEANS[true_states],
    scale=EMISSION_STD_DEVS[true_states],
)


# Fit several HMMs and keep the fit with the highest log-likelihood.
observations = simulated_returns.reshape(-1, 1)
best_model = None
best_log_likelihood = -np.inf

for seed in range(N_INITIALIZATIONS):
    candidate_model = GaussianHMM(
        n_components=NUM_STATES,
        covariance_type="diag",
        n_iter=2000,
        random_state=seed,
    )
    candidate_model.fit(observations)
    log_likelihood = candidate_model.score(observations)

    if log_likelihood > best_log_likelihood:
        best_log_likelihood = log_likelihood
        best_model = candidate_model

model = best_model


# Predict the hidden states with the fitted model.
predicted_states = model.predict(observations)


# HMM state labels are arbitrary, so align them before comparing states.
best_mapping = None
best_accuracy = -1.0

for permutation in permutations(range(NUM_STATES)):
    mapping = np.array(permutation)
    mapped_states = mapping[predicted_states]
    current_accuracy = accuracy_score(true_states, mapped_states)

    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_mapping = mapping

aligned_predicted_states = best_mapping[predicted_states]
state_order = np.argsort(best_mapping)


# Evaluate the state predictions.
classification_accuracy = accuracy_score(
    true_states,
    aligned_predicted_states,
)
state_confusion_matrix = confusion_matrix(
    true_states,
    aligned_predicted_states,
    labels=range(NUM_STATES),
)


# Align estimated parameters with the true state labels.
estimated_means = model.means_.ravel()[state_order]
estimated_variances = model.covars_.ravel()[state_order]
estimated_transition_matrix = model.transmat_[state_order][:, state_order]
estimated_initial_probs = model.startprob_[state_order]


# Save numerical results.
results = pd.DataFrame({
    "true_state": true_states,
    "predicted_state": aligned_predicted_states,
    "simulated_return": simulated_returns,
})
results.to_csv(RESULTS_DIR / "simulated_results.csv", index=False)

parameter_results = pd.DataFrame({
    "state": [STATE_NAMES[state] for state in range(NUM_STATES)],
    "true_mean": EMISSION_MEANS,
    "estimated_mean": estimated_means,
    "true_variance": EMISSION_STD_DEVS ** 2,
    "estimated_variance": estimated_variances,
})
parameter_results.to_csv(RESULTS_DIR / "parameter_results.csv", index=False)
np.savetxt(
    RESULTS_DIR / "true_transition_matrix.csv",
    TRANSITION_MATRIX,
    delimiter=",",
)
np.savetxt(
    RESULTS_DIR / "estimated_transition_matrix.csv",
    estimated_transition_matrix,
    delimiter=",",
)
np.savetxt(
    RESULTS_DIR / "confusion_matrix.csv",
    state_confusion_matrix,
    delimiter=",",
    fmt="%d",
)

pd.DataFrame({
    "metric": ["classification_accuracy", "log_likelihood"],
    "value": [classification_accuracy, best_log_likelihood],
}).to_csv(RESULTS_DIR / "metrics.csv", index=False)


create_plots(
    true_states=true_states,
    predicted_states=aligned_predicted_states,
    simulated_returns=simulated_returns,
    state_confusion_matrix=state_confusion_matrix,
    true_transition_matrix=TRANSITION_MATRIX,
    estimated_transition_matrix=estimated_transition_matrix,
    state_names=STATE_NAMES,
    figures_dir=FIGURES_DIR,
)
