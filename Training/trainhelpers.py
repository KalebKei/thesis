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
    else:
        print("Invalid encoding type: ", encoding)
    return f"{model_type}/{checkpoint_file}"
    

def save_hist(history, filename):
    with open(filename, "w") as f:
        print(history["train_loss"],file=f)
        print(history["train_acc"],file=f)
        print(history["val_loss"],file=f)
        print(history["val_acc"],file=f)
        print(history["layer1_fr"],file=f)
        print(history["layer2_fr"],file=f)
        print(history["output_fr"],file=f)
        print(history["layer1_spikes"],file=f)
        print(history["layer2_spikes"],file=f)
        print(history["output_spikes"],file=f)
        print(history["encoding"],file=f)
        print(history["type"],file=f)


def load_hist(filename):
    history = {
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

    with open(filename, "r") as f:
        train_loss = ast.literal_eval(f.readline().strip())
        history["train_loss"] = [float(x) for x in train_loss]
        train_acc = ast.literal_eval(f.readline().strip())
        history["train_acc"] = [float(x) for x in train_acc]
        val_loss = ast.literal_eval(f.readline().strip())
        history["val_loss"] = [float(x) for x in val_loss]
        val_acc = ast.literal_eval(f.readline().strip())
        history["val_acc"] = [float(x) for x in val_acc]
        layer1_fr = ast.literal_eval(f.readline().strip())
        history["layer1_fr"] = [int(x) for x in layer1_fr]
        layer2_fr = ast.literal_eval(f.readline().strip())
        history["layer2_fr"] = [int(x) for x in layer2_fr]
        output_fr = ast.literal_eval(f.readline().strip())
        history["output_fr"] = [int(x) for x in output_fr]
        layer1_spikes = ast.literal_eval(f.readline().strip())
        history["layer1_spikes"] = [int(x) for x in layer1_spikes]
        layer2_spikes = ast.literal_eval(f.readline().strip())
        history["layer2_spikes"] = [int(x) for x in layer2_spikes]
        output_spikes = ast.literal_eval(f.readline().strip())
        history["output_spikes"] = [int(x) for x in output_spikes]
        history["encoding"] = f.readline().strip()
        history["type"] = f.readline().strip()

    return history


def validate(
    model,
    val_loader,
    loss_fun,
    debug=False
):

    model.eval()

    running_loss = 0.0

    running_correct = 0
    running_total = 0

    running_layer1_fr = 0.0
    running_layer2_fr = 0.0
    running_output_fr = 0.0

    with torch.no_grad():

        for frames, labels in tqdm(
            val_loader,
            desc="Validation",
            leave=False
        ):

            frames = frames.float()

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

        print(
            f"\tOutput FR: {val_output_fr*100:.3f}%"
        )

    model.train()

    return {
        "loss": val_loss,
        "acc": val_acc,
        "layer1_fr": val_layer1_fr,
        "layer2_fr": val_layer2_fr,
        "output_fr": val_output_fr
    }

def train(model, train_loader, test_loader, optimizer, loss_fun, epochs, checkpoint_dir="ModelCheckpoints/", encoding="spike_train", model_type="SNN", history=None, debug=False):
    # call me thomas the way i be trainin

    model.train()

    # plot info
    if history is None:
        history = {
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


        for curr_batch, (frames, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {ep}", leave=False), start=1):
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


        val_metrics = validate(model, test_loader, loss_fun)

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

        # print
        if debug:
            print(f"\nEpoch {ep} Summary")
            print(f"\tLoss: {epoch_loss:.4f}")
            print(f"\tAccuracy: {epoch_acc*100:.2f}%")
            print(f"\tLayer1 Fire rate: {epoch_layer1_fr*100:.3f}%")
            print(f"\tLayer2 Fire rate: {epoch_layer2_fr*100:.3f}%")
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
        print(f"Saving history to {hist_path}") # debug
        save_hist(history=history, filename=hist_path)


    # go home
    return history

def plot_hist(history, epochs, filename_ext = ""):
    now = datetime.now()
    epoch_nums = range(1, epochs+1)
    
    plot_path = Path(f"Results/{match_dir(history['encoding'], history['type'])}")
    plot_path.mkdir(parents=True, exist_ok=True) # sanity check

    if(True):
        print(history)

    ##### plot val vs train acc
    plt.figure(figsize=(8,5))
    plt.plot(
        epoch_nums,
        [x * 100 for x in history["train_acc"]],
        marker='o',
        linewidth=2,
        label="Train"
    )
    plt.plot(
        epoch_nums,
        [x * 100 for x in history["val_acc"]],
        marker='s',
        linewidth=2,
        label="Validation"
    )
    plt.xticks(list(epoch_nums))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(
        f"{str(plot_path)}/{filename_ext}{now:%Y%m%d_%H%M}_{history['encoding']}_val_acc.png",
        dpi=300,
        bbox_inches="tight"
    )

    ##### plot loss
    plt.figure(figsize=(8,5))
    plt.plot(
        epoch_nums,
        history["train_loss"],
        marker='o',
        linewidth=2,
        label="Train"
    )
    plt.xticks(list(epoch_nums))
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    # plt.show()
    plt.savefig(
        f"{str(plot_path)}/{filename_ext}{now:%Y%m%d_%H%M}_{history['encoding']}_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    ##### plot fire rates
    plt.figure(figsize=(8,5))

    plt.plot(
        epoch_nums,
        [x * 100 for x in history["layer1_fr"]],
        marker='o',
        linewidth=2,
        label="Layer 1"
    )
    plt.plot(
        epoch_nums,
        [x * 100 for x in history["layer2_fr"]],
        marker='s',
        linewidth=2,
        label="Layer 2"
    )
    plt.plot(
        epoch_nums,
        [x * 100 for x in history["output_fr"]],
        marker='^',
        linewidth=2,
        label="Output"
    )
    plt.xticks(list(epoch_nums))
    plt.ylim(bottom=0)
    plt.title("Average Firing Rate by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Firing Rate (%)")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(
        f"{str(plot_path)}/{filename_ext}{now:%Y%m%d_%H%M}_{history['encoding']}_fr.png",
        dpi=300,
        bbox_inches="tight"
    )

    ##### plot spikes
    plt.figure(figsize=(8,5))
    plt.plot(
        epoch_nums,
        history["layer1_spikes"],
        marker='o',
        linewidth=2,
        label="Layer 1"
    )
    plt.plot(
        epoch_nums,
        history["layer2_spikes"],
        marker='s',
        linewidth=2,
        label="Layer 2"
    )
    plt.plot(
        epoch_nums,
        history["output_spikes"],
        marker='^',
        linewidth=2,
        label="Output"
    )
    plt.xticks(list(epoch_nums))
    plt.title("Total Spikes Generated per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Spike Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"{str(plot_path)}/{filename_ext}{now:%Y%m%d_%H%M}_{history['encoding']}_spike_counts.png",
        dpi=300,
        bbox_inches="tight"
    )
