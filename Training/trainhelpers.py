import sys
import os
import tonic
from tonic import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import snntorch as snn
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import ast
from pathlib import Path
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator

import numpy as np

def match_dir(encoding, model_type="SNN"):
    checkpoint_file = ""

    if(encoding == "spike_train"): 
        checkpoint_file = "SpikeTrain"
    elif(encoding == "voxel_grid"):
        checkpoint_file = "VoxelGrids"
    elif(encoding == "dct"):
        checkpoint_file = "DCT"
    elif(encoding == "trunc_dct"):
        checkpoint_file = "TruncatedDCT"
    elif(encoding == "aggr_dct"):
        checkpoint_file = "AggressiveDCT"
    elif(encoding == "interp_voxel_grid"):
        checkpoint_file = "InterpolatedVoxelGrids"
    else:
        print("Invalid encoding type: ", encoding)
    return f"{model_type}/{checkpoint_file}"

def is_model_three(model_type, dataset):
    if model_type == "SNN":
        if dataset == "NMNIST":
            return False
        else:
            return True
    else:
        if dataset == "NMNIST":
            return False
        else:
            return True
    

def save_hist(history, filename):
    with open(filename, "w") as f:
        print(history["dataset"], file=f)
        print(history["train_loss"],file=f)
        print(history["train_acc"],file=f)
        print(history["val_loss"],file=f)
        print(history["val_acc"],file=f)
        print(history["layer1_fr"],file=f)
        print(history["layer2_fr"],file=f)
        if(is_model_three(history["type"], history["dataset"])):
            print(history["layer3_fr"], file=f)
        print(history["output_fr"],file=f)
        print(history["layer1_spikes"],file=f)
        print(history["layer2_spikes"],file=f)
        if(is_model_three(history["type"], history["dataset"])):
            print(history["layer3_spikes"], file=f)
        print(history["output_spikes"],file=f)
        print(history["encoding"],file=f)
        print(history["type"],file=f)


def load_hist(filename, model_type, dataset):
    if not is_model_three(model_type, dataset):
        history = {
            "dataset": "",
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "layer1_fr": [],
            "layer2_fr": [],
            "output_fr": [],
            "layer1_spikes": [],
            "layer2_spikes": [],
            "output_spikes": [],
            "encoding": "",
            "type": ""
        }
    elif is_model_three(model_type, dataset):
        history = {
            "dataset": "",
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "layer1_fr": [],
            "layer2_fr": [],
            "layer3_fr": [],
            "output_fr": [],
            "layer1_spikes": [],
            "layer2_spikes": [],
            "layer3_spikes": [],
            "output_spikes": [],
            "encoding": "",
            "type": ""
        }

    with open(filename, "r") as f:
        history["dataset"] = f.readline().strip()
        train_loss = ast.literal_eval(f.readline().strip())
        history["train_loss"] = [float(x) for x in train_loss]
        train_acc = ast.literal_eval(f.readline().strip())
        history["train_acc"] = [float(x) for x in train_acc]
        val_loss = ast.literal_eval(f.readline().strip())
        history["val_loss"] = [float(x) for x in val_loss]
        val_acc = ast.literal_eval(f.readline().strip())
        history["val_acc"] = [float(x) for x in val_acc]
        layer1_fr = ast.literal_eval(f.readline().strip())
        history["layer1_fr"] = [float(x) for x in layer1_fr]
        layer2_fr = ast.literal_eval(f.readline().strip())
        history["layer2_fr"] = [float(x) for x in layer2_fr]
        if is_model_three(model_type, dataset):
            layer3_fr = ast.literal_eval(f.readline().strip())
            history["layer3_fr"] = [float(x) for x in layer3_fr]
        output_fr = ast.literal_eval(f.readline().strip())
        history["output_fr"] = [float(x) for x in output_fr]
        layer1_spikes = ast.literal_eval(f.readline().strip())
        history["layer1_spikes"] = [int(x) for x in layer1_spikes]
        layer2_spikes = ast.literal_eval(f.readline().strip())
        history["layer2_spikes"] = [int(x) for x in layer2_spikes]
        if is_model_three(model_type, dataset):
            layer3_spikes = ast.literal_eval(f.readline().strip())
            history["layer3_spikes"] = [int(x) for x in layer3_spikes]
        output_spikes = ast.literal_eval(f.readline().strip())
        history["output_spikes"] = [int(x) for x in output_spikes]
        history["encoding"] = f.readline().strip()
        history["type"] = model_type # technically not safe, but some models are already saved without this on the file, just in the name. so must be true to avoid errors


    return history


def validate(model, val_loader, loss_fun, model_type, device, dataset, debug=False):

    model.eval()
    model.to(device)

    running_loss = 0.0

    running_correct = 0
    running_total = 0

    running_layer1_fr = 0.0
    running_layer2_fr = 0.0
    if is_model_three(model_type, dataset):
        running_layer3_fr = 0.0
    running_output_fr = 0.0

    with torch.no_grad():

        for frames, labels in tqdm(
            val_loader,
            desc="Validation",
            leave=False
        ):
            frames = frames.float().to(device)
            labels = labels.to(device)

            spikes, stats = model(frames)

            logits = spikes.sum(dim=0)

            loss = loss_fun(logits, labels)

            predictions = logits.argmax(dim=1)

            correct = (
                predictions == labels
            ).sum().item()

            running_loss += loss.item()

            running_correct += correct

            running_total += labels.size(0)

            running_layer1_fr += stats["layer1fr"].item()
            running_layer2_fr += stats["layer2fr"].item()
            if is_model_three(model_type, dataset):
                running_layer3_fr += stats["layer3fr"].item()
            running_output_fr += stats["outputfr"].item()

    val_loss = running_loss / len(val_loader)

    val_acc = running_correct / running_total

    val_layer1_fr = (
        running_layer1_fr /
        len(val_loader)
    )

    val_layer2_fr = (
        running_layer2_fr /
        len(val_loader)
    )

    if is_model_three(model_type, dataset):
        val_layer3_fr = (
            running_layer3_fr /
            len(val_loader)
        )

    val_output_fr = (
        running_output_fr /
        len(val_loader)
    )

    if debug:

        print("\nValidation Summary")

        print(
            f"\tLoss: {val_loss:.4f}"
        )

        print(
            f"\tAccuracy: {val_acc*100:.2f}%"
        )

        print(
            f"\tLayer1 FR: {val_layer1_fr*100:.3f}%"
        )

        print(
            f"\tLayer2 FR: {val_layer2_fr*100:.3f}%"
        )

        if is_model_three(model_type, dataset):
            print(
                f"\tLayer3 FR: {val_layer3_fr*100:.3f}%"
            )

        print(
            f"\tOutput FR: {val_output_fr*100:.3f}%"
        )

    model.train()
    if is_model_three(model_type, dataset):
        return {
            "loss": val_loss,
            "acc": val_acc,
            "layer1_fr": val_layer1_fr,
            "layer2_fr": val_layer2_fr,
            "layer3_fr": val_layer3_fr,
            "output_fr": val_output_fr
        }
    return {
        "loss": val_loss,
        "acc": val_acc,
        "layer1_fr": val_layer1_fr,
        "layer2_fr": val_layer2_fr,
        "output_fr": val_output_fr
    }

def train(model, train_loader, test_loader, optimizer, loss_fun, epochs, device, checkpoint_dir="ModelCheckpoints/", encoding="spike_train", model_type="SNN", dataset="NMNIST", history=None, debug=False):
    # call me thomas the way i be trainin

    model.train()
    model.to(device)

    # plot info
    if history is None:
        if not is_model_three(model_type, dataset):
            history = {
                "dataset": dataset,
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": [],
                "layer1_fr": [],
                "layer2_fr": [],
                "output_fr": [],
                "layer1_spikes": [],
                "layer2_spikes": [],
                "output_spikes": [],
                "encoding": encoding,
                "type": model_type
            }
        elif is_model_three(model_type, dataset):
            history = {
                "dataset": dataset,
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": [],
                "layer1_fr": [],
                "layer2_fr": [],
                "layer3_fr": [],
                "output_fr": [],
                "layer1_spikes": [],
                "layer2_spikes": [],
                "layer3_spikes": [],
                "output_spikes": [],
                "encoding": encoding,
                "type": model_type
            }

        if(debug):
            print(f'Beginning training on {model_type} model with save location at {checkpoint_dir}')
    else:
        if(debug):
            print(f'Resuming training on {model_type} model with {history["val_acc"][-1]*100}% accuracy and {history["val_loss"][-1]:.2f} loss with save location at {checkpoint_dir}')

    for ep in tqdm(range(1, epochs + 1), desc=f"Training ({epochs} epochs)"):
        # data for plot
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        running_layer1_fr = 0.0
        running_layer2_fr = 0.0
        running_output_fr = 0.0

        running_layer1_spikes = 0
        running_layer2_spikes = 0
        running_output_spikes = 0

        if is_model_three(model_type, dataset):
            running_layer3_fr = 0.0
            running_layer3_spikes = 0.0


        for curr_batch, (frames, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {ep}", leave=False), start=1):
            frames = frames.to(device)
            labels = labels.to(device)
            # get batch frames
            frames = frames.float()

            optimizer.zero_grad()

            # what did the model see
            spikes, stats = model(frames)

            # raw
            logits = spikes.sum(dim=0)

            # lossing
            loss = loss_fun(logits, labels)

            # take it back now yall
            loss.backward()
            optimizer.step()

            # how'd we do
            predictions = logits.argmax(dim=1)
            correct = (predictions == labels).sum().item()

            # info for plots
            running_loss += loss.item()
            running_correct += correct
            running_total += labels.size(0)

            running_layer1_fr += stats["layer1fr"].item()
            running_layer2_fr += stats["layer2fr"].item()
            running_output_fr += stats["outputfr"].item()

            running_layer1_spikes += stats["layer1sp"].item()
            running_layer2_spikes += stats["layer2sp"].item()
            running_output_spikes += stats["outputsp"].item()
            
            if is_model_three(model_type, dataset): # need to get the third layer too if exists
                running_layer3_fr += stats["layer3fr"].item()
                running_layer3_spikes += stats["layer3sp"].item()


            if debug and curr_batch % 100 == 0:

                batch_acc = correct / labels.size(0)

                print(f"\nBatch {curr_batch}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Accuracy: {batch_acc*100:.2f}%")

        # get the info
        epoch_loss = (running_loss / len(train_loader))
        epoch_acc = (running_correct / running_total)
        epoch_layer1_fr = (running_layer1_fr / len(train_loader))
        epoch_layer2_fr = (running_layer2_fr / len(train_loader))
        epoch_output_fr = (running_output_fr / len(train_loader))
        epoch_layer1_spikes = running_layer1_spikes
        epoch_layer2_spikes = running_layer2_spikes
        epoch_output_spikes = running_output_spikes
        if is_model_three(model_type, dataset):
            epoch_layer3_spikes = running_layer3_spikes
            epoch_layer3_fr = (running_layer3_fr / len(train_loader))


        val_metrics = validate(model, test_loader, loss_fun, dataset=dataset, model_type=model_type, device=device)

        # update hist
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["layer1_fr"].append(epoch_layer1_fr)
        history["layer2_fr"].append(epoch_layer2_fr)
        history["output_fr"].append(epoch_output_fr)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["layer1_spikes"].append(epoch_layer1_spikes)
        history["layer2_spikes"].append(epoch_layer2_spikes)
        history["output_spikes"].append(epoch_output_spikes)
        if is_model_three(model_type, dataset):
            history["layer3_fr"].append(epoch_layer3_fr)
            history["layer3_spikes"].append(epoch_layer3_spikes)

        # print
        if debug:
            print(f"\nEpoch {ep} Summary")
            print(f"\tLoss: {epoch_loss:.4f}")
            print(f"\tAccuracy: {epoch_acc*100:.2f}%")
            print(f"\tValidation Accuracy: {val_metrics['acc']*100:.2f}%")
            print(f"\tLayer1 Fire rate: {epoch_layer1_fr*100:.3f}%")
            print(f"\tLayer2 Fire rate: {epoch_layer2_fr*100:.3f}%")
            if is_model_three(model_type, dataset):
                print(f"\tLayer3 Fire rate: {epoch_layer3_fr*100:.3f}%")
            print(f"\tOutput Fire rate: {epoch_output_fr*100:.3f}%")

        check_dir = Path(checkpoint_dir)
        check_dir.mkdir(parents=True, exist_ok=True) # sanity make sure it's there

        # save
        checkpoint_path = (
            f"{checkpoint_dir}/"
            f"{datetime.now():%Y%m%d_%H%M%S}_"
            f"checkpoint_epoch_{len(history['train_loss'])}.pt"
        )

        torch.save(
            {
                "epoch": ep,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "loss": epoch_loss
            },
            checkpoint_path
        )

        # save hist
        hist_path = (
            f"{checkpoint_dir}/"
            f"{datetime.now():%Y%m%d_%H%M%S}_hist_"
            f"checkpoint_epoch_{len(history['train_loss'])}.pt"
        )
        print(f"Saving history to {hist_path} with current validation accuracy of {val_metrics['acc']*100:.2f}%") # debug
        save_hist(history=history, filename=hist_path)


    # go home
    return history

def plot_hist(history, epochs, dataset, filename_ext="", debug=False):
    epoch_nums = range(1, epochs + 1)

    plot_path = Path(
        f"../Plots/Training/{dataset}/"
        f"{match_dir(history['encoding'], history['type'])}"
    )
    plot_path.mkdir(parents=True, exist_ok=True)

    if debug:
        print(history)

    prefix = f"{filename_ext}_{history['encoding']}"

    layers = [
        ("layer1", "Layer 1", "o"),
        ("layer2", "Layer 2", "s"),
    ]

    if is_model_three(history["type"], history["dataset"]):
        layers.append(("layer3", "Layer 3", "x"))

    layers.append(("output", "Output", "^"))

    def format_axis(ax, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=10, integer=True)
        )
        ax.grid(True)

    def plot_accuracy(ax):
        ax.plot(
            epoch_nums,
            [x * 100 for x in history["train_acc"]],
            marker="o",
            linewidth=2,
            label="Train",
        )
        ax.plot(
            epoch_nums,
            [x * 100 for x in history["val_acc"]],
            marker="s",
            linewidth=2,
            label="Validation",
        )
        format_axis(
            ax, "Training vs Validation Accuracy", "Accuracy (%)"
        )
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.legend()

    def plot_loss(ax):
        ax.plot(
            epoch_nums,
            history["train_loss"],
            marker="o",
            linewidth=2,
            label="Train",
        )
        format_axis(ax, "Training Loss", "Loss")
        ax.set_ylim(0, 2.5)
        ax.set_yticks(np.arange(0, 2.6, 0.25))

    def plot_firing_rates(ax):
        for key, label, marker in layers:
            ax.plot(
                epoch_nums,
                [x * 100 for x in history[f"{key}_fr"]],
                marker=marker,
                linewidth=2,
                label=label,
            )
        format_axis(
            ax, "Average Firing Rate by Epoch", "Firing Rate (%)"
        )
        ax.legend()

    def plot_spikes(ax):
        for key, label, marker in layers:
            ax.plot(
                epoch_nums,
                history[f"{key}_spikes"],
                marker=marker,
                linewidth=2,
                label=label,
            )
        format_axis(
            ax, "Total Spikes Generated per Epoch", "Spike Count"
        )
        ax.legend()

    plots = [
        ("val_acc", plot_accuracy),
        ("loss", plot_loss),
        ("fr", plot_firing_rates),
        ("spike_counts", plot_spikes),
    ]

    # Save the four individual plots.
    for suffix, draw_plot in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw_plot(ax)
        fig.tight_layout()
        fig.savefig(
            plot_path / f"{prefix}_{suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Save all four plots together in a 2×2 grid.
    fig, axes = plt.subplots(
        2, 2, figsize=(16, 10), constrained_layout=True
    )

    for ax, (_, draw_plot) in zip(axes.flat, plots):
        draw_plot(ax)

    fig.savefig(
        plot_path / f"{prefix}_combined.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


@torch.no_grad()
def evaluate_early_exit(model, test_loader, device, exit_steps=(10, 20, 30, 50)):
    exit_steps = sorted(set(exit_steps))

    if not exit_steps or exit_steps[0] < 1:
        raise ValueError("exit_steps must contain positive integers.")

    model.to(device)
    was_training = model.training
    model.eval()

    correct = {step: 0 for step in exit_steps}
    tied = {step: 0 for step in exit_steps}
    total = 0

    try:
        for frames, labels in test_loader:
            frames = frames.float().to(device)
            labels = labels.to(device)

            if exit_steps[-1] > frames.size(1):
                raise ValueError(
                    f"Requested step {exit_steps[-1]}, but input has "
                    f"only {frames.size(1)} steps."
                )

            # Process only through the largest requested stopping point.
            # Model output: [T, B, num_classes]
            spikes, _ = model(frames[:, :exit_steps[-1]])

            # Cumulative class spike counts at every step.
            cumulative_spikes = spikes.cumsum(dim=0)

            for step in exit_steps:
                scores = cumulative_spikes[step - 1]
                predictions = scores.argmax(dim=1)

                correct[step] += (
                    predictions == labels
                ).sum().item()

                # Count ambiguous predictions, including all-zero outputs.
                max_scores = scores.max(dim=1, keepdim=True).values
                tied[step] += (
                    (scores == max_scores).sum(dim=1) > 1
                ).sum().item()

            total += labels.size(0)

    finally:
        model.train(was_training)

    if total == 0:
        raise ValueError("test_loader contains no samples.")

    results = {}

    for step in exit_steps:
        results[step] = {
            "accuracy": 100.0 * correct[step] / total,
            "tie_rate": 100.0 * tied[step] / total,
            "samples": total,
        }

        print(
            f"Step {step:>2}: "
            f"accuracy = {results[step]['accuracy']:.2f}% | "
            f"tied predictions = {results[step]['tie_rate']:.2f}%"
        )

    return results

def plot_early_exit(results, history, filename_ext="", debug=False):
    if not results:
        raise ValueError("results contains no early-exit statistics.")

    if(debug):
        print("Printing early exit plots")
    steps = sorted(results)
    accuracies = [results[step]["accuracy"] for step in steps]
    tie_rates = [results[step]["tie_rate"] for step in steps]

    dataset = history["dataset"]
    model_type = history["type"]
    encoding = history["encoding"]

    plot_path = (
        Path("../Plots/EarlyExit")
        / str(dataset)
        / match_dir(encoding, model_type)
    )
    plot_path.mkdir(parents=True, exist_ok=True)

    if debug:
        print(
            f"Dataset: {dataset}\n"
            f"Model: {model_type}\n"
            f"Encoding: {encoding}\n"
            f"Results: {results}\n"
            f"Plot path: {plot_path}"
        )

    # Include all configuration identifiers to distinguish saved figures.
    prefix = "_".join(
        str(value)
        for value in (filename_ext, dataset, model_type, encoding)
        if value != ""
    )

    title = f"{dataset} | {model_type} | {encoding}"

    def format_axis(ax, plot_title, ylabel):
        ax.set_title(plot_title)
        ax.set_xlabel("Inference Timesteps")
        ax.set_ylabel(ylabel)
        ax.set_xticks(steps)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.grid(True, alpha=0.3)

    def plot_accuracy(ax):
        ax.plot(
            steps,
            accuracies,
            marker="o",
            linewidth=2,
            label="Accuracy",
        )
        ax.axhline(
            accuracies[-1],
            color="gray",
            linestyle="--",
            label=f"{steps[-1]}-step reference",
        )
        format_axis(
            ax,
            "Accuracy by Stopping Timestep",
            "Accuracy (%)",
        )
        ax.legend()

    def plot_tie_rate(ax):
        ax.plot(
            steps,
            tie_rates,
            marker="s",
            linewidth=2,
            color="tab:orange",
            label="Tied Predictions",
        )
        format_axis(
            ax,
            "Output Ambiguity by Stopping Timestep",
            "Tied Predictions (%)",
        )
        ax.legend()

    plots = [
        ("accuracy", plot_accuracy),
        ("tie_rate", plot_tie_rate),
    ]

    saved_paths = []

    # Save individual plots.
    for suffix, draw_plot in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw_plot(ax)
        fig.suptitle(title)
        fig.tight_layout()

        output_path = plot_path / f"{prefix}_{suffix}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    # Save both plots together.
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        constrained_layout=True,
    )
    fig.suptitle(title)

    for ax, (_, draw_plot) in zip(axes, plots):
        draw_plot(ax)

    output_path = plot_path / f"{prefix}_combined.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(output_path)

    return saved_paths