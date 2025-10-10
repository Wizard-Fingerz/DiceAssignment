import json
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

# For saving table as image
try:
    import dataframe_image as dfi
    DFI_AVAILABLE = True
except ImportError:
    DFI_AVAILABLE = False

# Path to the output folder from your training
checkpoints_dir = "./distil_finetuned_freshers"
pattern = os.path.join(checkpoints_dir, "checkpoint-*/trainer_state.json")
log_files = sorted(glob.glob(pattern))

if not log_files:
    print(f"Error: No trainer_state.json files found in '{checkpoints_dir}'")
    print("Please ensure the training has completed and the trainer_state.json files exist in the checkpoint directories.")
else:
    all_metrics = []
    for log_file in log_files:
        checkpoint_name = os.path.basename(os.path.dirname(log_file))
        with open(log_file, "r") as f:
            trainer_state = json.load(f)
        log_history = trainer_state.get("log_history", [])
        for entry in log_history:
            entry = entry.copy()  # avoid mutating the original
            entry["checkpoint"] = checkpoint_name
            all_metrics.append(entry)
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        print(df_metrics)
        df_metrics.to_csv("task2_training_results_all_checkpoints.csv", index=False)
        print("\nSaved results to 'task2_training_results_all_checkpoints.csv'")

        # Plotting: Loss and Accuracy over Steps
        if "step" in df_metrics.columns:
            # Plot training loss
            if "loss" in df_metrics.columns:
                plt.figure(figsize=(10, 5))
                plt.plot(df_metrics["step"], df_metrics["loss"], label="Training Loss", marker='o')
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title("Training Loss over Steps")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("task2_training_loss.png")
                print("Saved training loss plot as 'task2_training_loss.png'")
                plt.close()

            # Plot eval loss if available
            if "eval_loss" in df_metrics.columns:
                plt.figure(figsize=(10, 5))
                plt.plot(df_metrics["step"], df_metrics["eval_loss"], label="Eval Loss", marker='o', color='orange')
                plt.xlabel("Step")
                plt.ylabel("Eval Loss")
                plt.title("Eval Loss over Steps")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("task2_eval_loss.png")
                print("Saved eval loss plot as 'task2_eval_loss.png'")
                plt.close()

            # Plot eval accuracy if available
            if "eval_accuracy" in df_metrics.columns:
                plt.figure(figsize=(10, 5))
                plt.plot(df_metrics["step"], df_metrics["eval_accuracy"], label="Eval Accuracy", marker='o', color='green')
                plt.xlabel("Step")
                plt.ylabel("Eval Accuracy")
                plt.title("Eval Accuracy over Steps")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("task2_eval_accuracy.png")
                print("Saved eval accuracy plot as 'task2_eval_accuracy.png'")
                plt.close()

        # Save table as image if possible
        if DFI_AVAILABLE:
            try:
                dfi.export(df_metrics, "task2_training_results_table.png")
                print("Saved results table as image: 'task2_training_results_table.png'")
            except Exception as e:
                print(f"Failed to save table as image: {e}")
        else:
            print("dataframe_image is not installed. To save the table as an image, install it via 'pip install dataframe_image'.")

    else:
        print("No log history found in any checkpoint.")
