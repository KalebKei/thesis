import pandas as pd
from trainhelpers import load_hist
import sys
import os
import tonic
import torch
from torch.utils.data import DataLoader, random_split
import gc
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.shared import Inches, Pt
from pathlib import Path
import trainhelpers as th

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.snn_baseline import SNNModel_Gesture
from Models.snn_wavelet import WaveletModel_Gesture
from Models.snn_wavelet_integrated import IntegratedWaveletBlockModel_Gesture
import Encodings.gestureencodings as encodings

to_docx = True

# SNN
snn_spike_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/SNN/SpikeTrain/20260914_203216_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
snn_event_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/SNN/EventBins/20261009_081939_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
snn_dct_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/SNN/DCT/20261010_094257_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
snn_trunc_dct_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/SNN/TruncatedDCT/20261011_095737_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
snn_voxel_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/SNN/VoxelGrids/20261012_170406_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")

snn_spike_model_path= "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/SNN/SpikeTrain/20260914_203216_checkpoint_epoch_500.pt"
snn_event_model_path= "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/SNN/EventBins/20261009_081939_checkpoint_epoch_500.pt"
snn_dct_model_path= "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/SNN/DCT/20261010_094257_checkpoint_epoch_500.pt"
snn_trunc_dct_model_path= "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/SNN/TruncatedDCT/20261011_095737_checkpoint_epoch_500.pt"
snn_voxel_model_path= "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/SNN/VoxelGrids/20261012_170406_checkpoint_epoch_500.pt"

# FrontEndWaveletSNN
wavelet_spike_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/FrontEndWaveletSNN/SpikeTrain/20261013_184132_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
wavelet_event_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/FrontEndWaveletSNN/EventBins/20261014_175027_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
wavelet_dct_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/FrontEndWaveletSNN/DCT/20261015_172542_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
wavelet_trunc_dct_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/FrontEndWaveletSNN/TruncatedDCT/20261016_174113_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
wavelet_voxel_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/FrontEndWaveletSNN/VoxelGrids/20261017_162409_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")

wavelet_spike_model_path= "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/FrontEndWaveletSNN/SpikeTrain/20261013_184132_checkpoint_epoch_500.pt"
wavelet_event_model_path= "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/FrontEndWaveletSNN/EventBins/20261014_175027_checkpoint_epoch_500.pt"
wavelet_dct_model_path= "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/FrontEndWaveletSNN/DCT/20261015_172542_checkpoint_epoch_500.pt"
wavelet_trunc_dct_model_path= "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/FrontEndWaveletSNN/TruncatedDCT/20261016_174113_checkpoint_epoch_500.pt"
wavelet_voxel_model_path= "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/FrontEndWaveletSNN/VoxelGrids/20261017_162409_checkpoint_epoch_500.pt"


# IntegratedWaveletSNN
integrated_spike_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/IntegratedWaveletSNN/SpikeTrain/20261030_015324_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
integrated_event_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/IntegratedWaveletSNN/EventBins/20261105_192817_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
integrated_dct_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/IntegratedWaveletSNN/DCT/20261030_162000_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
integrated_trunc_dct_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/IntegratedWaveletSNN/TruncatedDCT/20261022_222435_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")
integrated_voxel_history = load_hist("/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/IntegratedWaveletSNN/VoxelGrids/20261031_032132_hist_checkpoint_epoch_500.pt", model_type="SNN", dataset="DVSGesture")

integrated_spike_model_path = "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/IntegratedWaveletSNN/SpikeTrain/20261030_015324_checkpoint_epoch_500.pt"
integrated_event_model_path = "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/IntegratedWaveletSNN/EventBins/20261105_192817_checkpoint_epoch_500.pt"
integrated_dct_model_path = "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/IntegratedWaveletSNN/DCT/20261030_162000_checkpoint_epoch_500.pt"
integrated_trunc_dct_model_path = "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/IntegratedWaveletSNN/TruncatedDCT/20261022_222435_checkpoint_epoch_500.pt"
integrated_voxel_model_path = "/home/kaleb/Thesis/thesis/SAVEDFINALWEIGHTS/DVSGesture/IntegratedWaveletSNN/VoxelGrids/20261031_032132_checkpoint_epoch_500.pt"




models_hist = {
    "SNN": {
        "spike_train": snn_spike_history,
        "event_bins": snn_event_history,
        "dct": snn_dct_history,
        "dct_truncated": snn_trunc_dct_history,
        "voxel_grids": snn_voxel_history
    },
    "FrontEndWaveletSNN": {
        "spike_train": wavelet_spike_history,
        "event_bins": wavelet_event_history,
        "dct": wavelet_dct_history,
        "dct_truncated": wavelet_trunc_dct_history,
        "voxel_grids": wavelet_voxel_history
    },
    "IntegratedWaveletSNN": {
        "spike_train": integrated_spike_history,
        "event_bins": integrated_event_history,
        "dct": integrated_dct_history,
        "dct_truncated": integrated_trunc_dct_history,
        "voxel_grids": integrated_voxel_history
    },
}


def export_tables_to_word(tables, filename="Results_Tables.docx"):
    """
    tables: {"Table title": dataframe, ...}
    """
    document = Document()

    # Landscape orientation gives wide tables more room.
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = (
        section.page_height,
        section.page_width,
    )
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)

    document.styles["Normal"].font.size = Pt(9)

    for index, (title, dataframe) in enumerate(tables.items()):
        if index:
            document.add_page_break()

        document.add_heading(title, level=1)

        # Include Model, Encoding, and Time Steps index columns.
        dataframe = dataframe.reset_index()

        table = document.add_table(
            rows=1,
            cols=len(dataframe.columns),
        )
        table.style = "Table Grid"

        for cell, column in zip(table.rows[0].cells, dataframe.columns):
            cell.text = str(column)
            for run in cell.paragraphs[0].runs:
                run.bold = True

        for values in dataframe.itertuples(index=False, name=None):
            cells = table.add_row().cells

            for cell, value in zip(cells, values):
                if pd.isna(value):
                    cell.text = "—"
                elif isinstance(value, float):
                    cell.text = f"{value:,.4f}"
                else:
                    cell.text = str(value)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(output_path))

    print(f"Saved: {output_path.resolve()}")

def compare_early_exit(models_early, full_steps=50):
    """
    models = {
        model_name: {
            encoding_name: {
                time_steps: {
                    "accuracy": float,       # Fraction: 0–1
                    "ms_per_sample": float,  # Optional
                    "spikes_per_sample": float,  # Optional
                }
            }
        }
    }
    """
    rows = []

    for model, encodings in models_early.items():
        for encoding, results in encodings.items():
            if full_steps not in results:
                raise ValueError(
                    f"{model}/{encoding} is missing "
                    f"the {full_steps}-step reference."
                )

            reference = results[full_steps]["accuracy"]

            for steps, metrics in sorted(results.items()):
                row = {
                    "Model": model,
                    "Encoding": encoding,
                    "Time Steps": steps,
                    "Accuracy (%)": metrics["accuracy"],
                    "Accuracy Change (pp)": metrics["accuracy"] - reference,
                    "Steps Used (%)": steps / full_steps * 100,
                }

                if "ms_per_sample" in metrics:
                    row["Time per Sample (ms)"] = (
                        metrics["ms_per_sample"]
                    )

                if "spikes_per_sample" in metrics:
                    row["Spikes per Sample"] = (
                        metrics["spikes_per_sample"]
                    )

                rows.append(row)

    if not rows:
        raise ValueError("No early-exit results provided.")

    table = pd.DataFrame(rows).set_index(
        ["Model", "Encoding", "Time Steps"]
    )

    print("\nEarly-Exit Comparison:")
    print(table.round(4).to_string(na_rep="—"))

    return table

def compare_histories(models):
    """
    models: {model_name: {encoding_name: history}}

    All histories must belong to the same dataset.
    Uses final-epoch metrics, plus best validation accuracy.
    """
    runs = [
        (model, encoding, history)
        for model, encodings in models.items()
        for encoding, history in encodings.items()
    ]
    if not runs:
        raise ValueError("No histories provided.")

    datasets = {history["dataset"] for _, _, history in runs}

    if len(datasets) != 1:
        raise ValueError("All histories must use the same dataset.")

    performance_rows = []
    activity_rows = []

    for model, encoding, history in runs:
        performance_rows.append({
            "Model": model,
            "Encoding": encoding,
            "Epochs": len(history["train_acc"]),
            "Train Acc (%)": history["train_acc"][-1] * 100,
            "Val Acc (%)": history["val_acc"][-1] * 100,
            "Best Val Acc (%)": max(history["val_acc"]) * 100,
            "Train Loss": history["train_loss"][-1],
            "Val Loss": history["val_loss"][-1],
        })

        activity = {
            "Model": model,
            "Encoding": encoding,
        }

        layers = [
            ("layer1", "L1"),
            ("layer2", "L2"),
        ]

        if "layer3_fr" in history:
            layers.append(("layer3", "L3"))

        layers.append(("output", "Output"))

        for key, label in layers:
            activity[f"{label} FR (%)"] = (
                history[f"{key}_fr"][-1] * 100
            )
            activity[f"{label} Spikes"] = (
                history[f"{key}_spikes"][-1]
            )

        activity_rows.append(activity)

    performance = pd.DataFrame(performance_rows).set_index(
        ["Model", "Encoding"]
    )
    activity = pd.DataFrame(activity_rows).set_index(
        ["Model", "Encoding"]
    )

    print(f"\nDataset: {next(iter(datasets))}")

    print("\nPerformance:")
    print(performance.round(4).to_string())

    print("\nFinal-epoch firing rates and spike counts:")
    print(activity.round(4).to_string(na_rep="—"))

    return performance, activity


def evaluate_all_early_exit(
    models,
    encoding_transforms,
    device,
    batch_size=32,
    exit_steps=(10, 20, 30, 50),
    save_to="../Datasets/DVSGESTURE/",
):
    device = torch.device(device)
    results = {name: {} for name in models}

    # Validate transforms before starting evaluation.
    encodings = list(dict.fromkeys(
        encoding
        for model_encodings in models.values()
        for encoding in model_encodings
    ))

    for encoding in encodings:
        if encoding not in encoding_transforms:
            raise ValueError(f"Missing transform for {encoding}")

    # Keep inactive models on CPU.
    for model_encodings in models.values():
        for model in model_encodings.values():
            model.to("cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    for encoding in encodings:
        # One dataset/loader for this encoding, shared across models.
        test_dataset = tonic.datasets.DVSGesture(
            save_to=save_to,
            train=False,
            transform=encoding_transforms[encoding],
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        try:
            for model_name, model_encodings in models.items():
                if encoding not in model_encodings:
                    continue

                print(f"\nEvaluating {model_name} / {encoding}")

                model = model_encodings[encoding]
                was_training = model.training

                try:
                    model.to(device)
                    model.eval()

                    with torch.no_grad():
                        results[model_name][encoding] = (
                            th.evaluate_early_exit(
                                model=model,
                                test_loader=test_loader,
                                device=device,
                                exit_steps=exit_steps,
                            )
                        )
                finally:
                    model.to("cpu")
                    model.train(was_training)

                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        finally:
            del test_loader, test_dataset
            gc.collect()

    return results



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prep the models
#map_location=torch.device('cuda'), weights_only=True))

# SNN spike models
snn_spike_model = SNNModel_Gesture().to(device)
checkpoint = torch.load(snn_spike_model_path,map_location=device,weights_only=True)
snn_spike_model.load_state_dict(checkpoint["model_state_dict"])
snn_spike_model.eval()

snn_event_model = SNNModel_Gesture().to(device)
checkpoint = torch.load(snn_event_model_path,map_location=device,weights_only=True)
snn_event_model.load_state_dict(checkpoint["model_state_dict"])
snn_event_model.eval()

snn_dct_model = SNNModel_Gesture().to(device)
checkpoint = torch.load(snn_dct_model_path,map_location=device,weights_only=True)
snn_dct_model.load_state_dict(checkpoint["model_state_dict"])
snn_dct_model.eval()

snn_trunc_dct_model = SNNModel_Gesture().to(device)
checkpoint = torch.load(snn_trunc_dct_model_path,map_location=device,weights_only=True)
snn_trunc_dct_model.load_state_dict(checkpoint["model_state_dict"])
snn_trunc_dct_model.eval()

snn_voxel_model = SNNModel_Gesture().to(device)
checkpoint = torch.load(snn_voxel_model_path,map_location=device,weights_only=True)
snn_voxel_model.load_state_dict(checkpoint["model_state_dict"])
snn_voxel_model.eval()

# Front End Wavelet SNN
wavelet_spike_model = WaveletModel_Gesture().to(device)
checkpoint = torch.load(wavelet_spike_model_path,map_location=device,weights_only=True)
wavelet_spike_model.load_state_dict(checkpoint["model_state_dict"])
wavelet_spike_model.eval()

wavelet_event_model = WaveletModel_Gesture().to(device)
checkpoint = torch.load(wavelet_event_model_path,map_location=device,weights_only=True)
wavelet_event_model.load_state_dict(checkpoint["model_state_dict"])
wavelet_event_model.eval()

wavelet_dct_model = WaveletModel_Gesture().to(device)
checkpoint = torch.load(wavelet_dct_model_path,map_location=device,weights_only=True)
wavelet_dct_model.load_state_dict(checkpoint["model_state_dict"])
wavelet_dct_model.eval()

wavelet_trunc_dct_model = WaveletModel_Gesture().to(device)
checkpoint = torch.load(wavelet_trunc_dct_model_path,map_location=device,weights_only=True)
wavelet_trunc_dct_model.load_state_dict(checkpoint["model_state_dict"])
wavelet_trunc_dct_model.eval()

wavelet_voxel_model = WaveletModel_Gesture().to(device)
checkpoint = torch.load(wavelet_voxel_model_path,map_location=device,weights_only=True)
wavelet_voxel_model.load_state_dict(checkpoint["model_state_dict"])
wavelet_voxel_model.eval()

# Integrated Wavelet
integrated_spike_model = IntegratedWaveletBlockModel_Gesture().to(device)
checkpoint = torch.load(integrated_spike_model_path,map_location=device,weights_only=True)
integrated_spike_model.load_state_dict(checkpoint["model_state_dict"])
integrated_spike_model.eval()

integrated_event_model = IntegratedWaveletBlockModel_Gesture().to(device)
checkpoint = torch.load(integrated_event_model_path,map_location=device,weights_only=True)
integrated_event_model.load_state_dict(checkpoint["model_state_dict"])
integrated_event_model.eval()

integrated_dct_model = IntegratedWaveletBlockModel_Gesture().to(device)
checkpoint = torch.load(integrated_dct_model_path,map_location=device,weights_only=True)
integrated_dct_model.load_state_dict(checkpoint["model_state_dict"])
integrated_dct_model.eval()

integrated_trunc_dct_model = IntegratedWaveletBlockModel_Gesture().to(device)
checkpoint = torch.load(integrated_trunc_dct_model_path,map_location=device,weights_only=True)
integrated_trunc_dct_model.load_state_dict(checkpoint["model_state_dict"])
integrated_trunc_dct_model.eval()

integrated_voxel_model = IntegratedWaveletBlockModel_Gesture().to(device)
checkpoint = torch.load(integrated_voxel_model_path,map_location=device,weights_only=True)
integrated_voxel_model.load_state_dict(checkpoint["model_state_dict"])
integrated_voxel_model.eval()


# Define and calculate (this is computationally expensive brace yourselfs)
models_early = {
    "SNN": {
        "spike_train": snn_spike_model,
        "event_bins": snn_event_model,
        "dct": snn_dct_model,
        "dct_truncated": snn_trunc_dct_model,
        "voxel_grids": snn_voxel_model
    },
    "FrontEndWaveletSNN": {
        "spike_train": wavelet_spike_model,
        "event_bins": wavelet_event_model,
        "dct": wavelet_dct_model,
        "dct_truncated": wavelet_trunc_dct_model,
        "voxel_grids": wavelet_voxel_model
    },
    "IntegratedWaveletSNN": {
        "spike_train": integrated_spike_model,
        "event_bins": integrated_event_model,
        "dct": integrated_dct_model,
        "dct_truncated": integrated_trunc_dct_model,
        "voxel_grids": integrated_voxel_model
    },
}

encoding_transforms = {
    "spike_train": encodings.spiketrain_transform,
    "event_bins": encodings.voxel_grids_transform,
    "dct": encodings.dct_transform,
    "dct_truncated": encodings.truncated_dct_transform,
    "voxel_grids": encodings.interpolated_voxel_grid_transform,
}


performance, activity = compare_histories(models_hist)

early_results = evaluate_all_early_exit(
    models=models_early,
    encoding_transforms=encoding_transforms,
    device=device,
    batch_size=32,
)

early_exit_table = compare_early_exit(early_results)

if(to_docx):
    export_tables_to_word(
    {
        "Training Performance by Model and Encoding": performance,
        "Firing Rates and Spike Counts": activity,
    },
    filename="../Plots/ComparisonCharts/DVSGesture_Training_Comparison.docx",
    )

    model_tables = {
        f"Early-Exit Comparison — {model_name}": (
            early_exit_table.xs(model_name, level="Model")
        )
        for model_name in early_exit_table.index
            .get_level_values("Model").unique()
    }
    export_tables_to_word(
        model_tables,
        filename="../Plots/ComparisonCharts/DVSGesture_Early_Exit_Comparison.docx",
    )
    