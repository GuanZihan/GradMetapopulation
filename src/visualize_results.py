import matplotlib.pyplot as plt
import os

def plot_predictions(target, predictions, rmse, rmse_test, state_idx, args, N_lag=0):
    figdir = "Figure-Prediction"
    os.makedirs(figdir, exist_ok=True)

    # Style
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curves
    ax.plot(target, label="Ground Truth", color="black", linewidth=2, alpha=0.8)
    ax.plot(range(N_lag, len(target)), predictions, label="Predictions", 
            color="royalblue", linewidth=2, linestyle="--", marker="o", markersize=4)

    # Labels and title
    ax.set_xlabel("Timestamp", fontsize=14)
    ax.set_ylabel("# Infections", fontsize=14)
    ax.set_title(f"Predictions vs Ground Truth (State {state_idx})", fontsize=16, fontweight="bold", pad=15)

    # # Annotate RMSE
    # ax.text(0.02, 0.95,
    #         f"Train RMSE: {rmse:.3f}\nTest RMSE: {rmse_test:.3f}",
    #         transform=ax.transAxes, fontsize=12,
    #         verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", 
    #                                            facecolor="white", alpha=0.7))

    # Legend and grid
    ax.legend(fontsize=12, loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Layout & save
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"State_{state_idx}_{args.date}_{args.note}.png"), dpi=300)
    plt.close(fig)


def plot_predictions_nowcasting(revisions, target, predictions, rmse, rmse_test, state_idx, args, N_lag=0):
    figdir = "Figure-Prediction"
    os.makedirs(figdir, exist_ok=True)

    # Save arrays for reproducibility
    with open(f"target_{N_lag}.txt", "w") as f:
        f.write(str(target.tolist())[1:-1])
        f.write("\n\n")
        f.write(str(predictions.tolist())[1:-1])

    # Style
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ground truth
    ax.plot(target, label="Ground Truth", color="black", linewidth=2, alpha=0.9)

    # Plot predictions
    ax.plot(range(N_lag, len(target)), predictions, 
            label="Predictions", color="royalblue", linestyle="--", 
            linewidth=2, marker="o", markersize=4)

    # Plot revisions (as faint background curves)
    ax.plot(revisions, color="green", alpha=0.3, linewidth=2)

    # Labels & title
    ax.set_xlabel("Timestamp", fontsize=14)
    ax.set_ylabel("# Infections", fontsize=14)
    ax.set_title(f"Nowcasting Predictions (State {state_idx})", fontsize=16, fontweight="bold", pad=15)

    # Annotate RMSE values
    ax.text(0.02, 0.95,
            f"Train RMSE: {rmse:.3f}\nTest RMSE: {rmse_test:.3f}",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7))

    # Legend
    ax.legend(fontsize=12, loc="best", frameon=True)

    # Grid and layout
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    # Save high-quality
    fig.savefig(os.path.join(figdir, f"{args.note}.png"), dpi=300)
    plt.close(fig)


def plot_losses(losses, params, args, loss_type="gradmeta"):
    disease = params['disease']
    FIGPATH = f'./Figures-Loss/{disease}/'
    os.makedirs(FIGPATH, exist_ok=True)

    # Create figure
    plt.style.use("seaborn-v0_8-darkgrid")  # clean modern style
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot losses
    ax.plot(losses, color="royalblue", linewidth=2, marker="o", markersize=4, label="Training Loss")

    # Labels & title
    ax.set_title(f"Training Loss Curve ({disease})", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)

    # Grid, ticks, legend
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)

    # Tight layout for clean saving
    fig.tight_layout()
    fig.savefig(os.path.join(FIGPATH, f'losses_curve_{loss_type}.png'), dpi=300)
    plt.close(fig)