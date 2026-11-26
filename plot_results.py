import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_plots(
    true_states,
    predicted_states,
    simulated_returns,
    state_confusion_matrix,
    true_transition_matrix,
    estimated_transition_matrix,
    state_names,
    figures_dir,
):
    """Create and save all simulation figures."""
    num_states = len(state_names)
    time = np.arange(len(simulated_returns))
    colors = ["tab:green", "tab:red", "tab:gray"]
    labels = list(state_names.values())

    plt.figure(figsize=(12, 4))
    for state in range(num_states):
        mask = true_states == state
        plt.scatter(time[mask], simulated_returns[mask], s=8,
                    color=colors[state], label=state_names[state])
    plt.xlabel("Observation")
    plt.ylabel("Return")
    plt.legend()
    _save_plot(figures_dir, "returns_by_true_regime.png",
               "Simulated Returns by True Regime")

    plt.figure(figsize=(12, 4))
    for state in range(num_states):
        mask = predicted_states == state
        plt.scatter(time[mask], simulated_returns[mask], s=8,
                    color=colors[state], label=state_names[state])
    plt.xlabel("Observation")
    plt.ylabel("Return")
    plt.legend()
    _save_plot(figures_dir, "returns_by_predicted_regime.png",
               "Simulated Returns by Predicted Regime")

    plt.figure(figsize=(12, 3))
    plt.step(time, true_states, where="post", label="True", linewidth=1)
    plt.step(time, predicted_states, where="post", label="Predicted",
             linewidth=1, alpha=0.8)
    plt.yticks(range(num_states), labels)
    plt.xlabel("Observation")
    plt.ylabel("Regime")
    plt.legend()
    _save_plot(figures_dir, "true_vs_predicted_states.png",
               "True vs. Predicted Regimes")

    plt.figure(figsize=(5, 4))
    plt.imshow(state_confusion_matrix, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xticks(range(num_states), labels)
    plt.yticks(range(num_states), labels)
    plt.xlabel("Predicted regime")
    plt.ylabel("True regime")
    for row in range(num_states):
        for column in range(num_states):
            plt.text(column, row, state_confusion_matrix[row, column],
                     ha="center", va="center")
    _save_plot(figures_dir, "confusion_matrix.png", "Confusion Matrix")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, matrix, title in zip(
        axes,
        [true_transition_matrix, estimated_transition_matrix],
        ["True Transition Matrix", "Estimated Transition Matrix"],
    ):
        image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xticks(range(num_states), labels)
        axis.set_yticks(range(num_states), labels)
        for row in range(num_states):
            for column in range(num_states):
                axis.text(column, row, f"{matrix[row, column]:.2f}",
                          ha="center", va="center", color="white")
    fig.colorbar(image, ax=axes, label="Probability")
    fig.tight_layout()
    fig.savefig(figures_dir / "transition_matrices.png", dpi=150)
    plt.close(fig)


def _save_plot(figures_dir, filename, title):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150)
    plt.close()
