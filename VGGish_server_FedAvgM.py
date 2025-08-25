# server.py (Using FedAvgM with full summary output)
import flwr as fl
import tensorflow as tf
import numpy as np
import joblib
import os
import time
import random
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import classification_report, f1_score

# --- Configuration ---
NUM_ROUNDS = 44
SERVER_ADDRESS = "0.0.0.0:8080"
EXPECTED_CLIENTS = 3
TEST_NPZ_FILE = 'vggish_test_features_labels.npz'
SCALER_FILE = 'global_scaler.pkl'

INPUT_DIM = 128
UNITS1 = 128
UNITS2 = 64
UNITS3 = 32
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

def build_model(input_shape_dim=INPUT_DIM):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape_dim,)),
        tf.keras.layers.Dense(UNITS1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(UNITS2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(UNITS3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def fit_metrics_aggregation_fn(results: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
    total_examples = sum([num_examples for num_examples, _ in results])
    if total_examples == 0:
        return {}

    # --- NEW: Loop through results to print individual client metrics ---
    print(f"--- Individual Client Fit Metrics (Round) ---")
    for i, (num_examples, metrics) in enumerate(results):
        client_id_placeholder = "Client " + str(i) # Assuming client IDs are sequential or not important
        local_acc = metrics.get("local_accuracy", 0.0)
        local_loss = metrics.get("local_loss", 0.0)
        local_epochs = metrics.get("local_epochs_run", 0.0)
        
        print(f" {client_id_placeholder}: Examples={num_examples}, Acc={local_acc:.4f}, Loss={local_loss:.4f}, Epochs={local_epochs:.1f}")
    
    # --- Existing: Calculate and print aggregated metrics ---
    avg_local_accuracy = sum(num_examples * metrics.get("local_accuracy", 0.0) for num_examples, metrics in results) / total_examples
    avg_local_loss = sum(num_examples * metrics.get("local_loss", 0.0) for num_examples, metrics in results) / total_examples
    avg_local_epochs = sum(num_examples * metrics.get("local_epochs_run", 0.0) for num_examples, metrics in results) / total_examples
    
    print(f"--- Aggregated Client Fit Metrics (Round) ---")
    print(f" Avg. Local Acc: {avg_local_accuracy:.4f}, Avg. Local Loss: {avg_local_loss:.4f}, Avg. Local Epochs: {avg_local_epochs:.1f}")

    return {
        "avg_local_accuracy": avg_local_accuracy,
        "avg_local_loss": avg_local_loss,
        "avg_local_epochs": avg_local_epochs
    }

def get_evaluate_fn(test_x, test_y):
    model = build_model()
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        if not test_x.size or not test_y.size:
            return 0.0, {"accuracy": 0.0}
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
        y_pred_proba = model.predict(test_x, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        report_dict = classification_report(test_y, y_pred, output_dict=True, digits=4, zero_division=0)
        f1_class_0 = report_dict.get('0', {}).get('f1-score', 0.0)
        f1_class_1 = report_dict.get('1', {}).get('f1-score', 0.0)
        f1_weighted = f1_score(test_y, y_pred, average='weighted', zero_division=0)
        print(f"--- Server-Side Eval (Round {server_round}) -> Loss: {loss:.4f}, Acc: {accuracy:.4f}, F1(W): {f1_weighted:.4f} ---")
        return loss, {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_class_0": f1_class_0,
            "f1_class_1": f1_class_1
        }
    return evaluate

def on_fit_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration to the client for the current round."""
    config = {
        "local_epochs": 100, 
        "batch_size": 32,
        "server_round": server_round
    }
    return config


def fit_metrics_aggregation_fn_OLD(results: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
    total_examples = sum([num_examples for num_examples, _ in results])
    if total_examples == 0:
        return {}
    avg_local_accuracy = sum(num_examples * metrics.get("local_accuracy", 0.0) for num_examples, metrics in results) / total_examples
    avg_local_loss = sum(num_examples * metrics.get("local_loss", 0.0) for num_examples, metrics in results) / total_examples
    avg_local_epochs = sum(num_examples * metrics.get("local_epochs_run", 0.0) for num_examples, metrics in results) / total_examples
    print(f"--- Aggregated Client Fit Metrics (Round) ---")
    print(f" Avg. Local Acc: {avg_local_accuracy:.4f}, Avg. Local Loss: {avg_local_loss:.4f}, Avg. Local Epochs: {avg_local_epochs:.1f}")
    return {
        "avg_local_accuracy": avg_local_accuracy,
        "avg_local_loss": avg_local_loss,
        "avg_local_epochs": avg_local_epochs
    }

if __name__ == "__main__":
    print("Starting Flower Server with FedAvgM Strategy...")

    # Load Scaler
    try:
        scaler = joblib.load(SCALER_FILE)
        print(f"Loaded global scaler from {SCALER_FILE}")
    except Exception as e:
        print(f"Error loading scaler '{SCALER_FILE}': {e}")
        exit()

    # Load Test Data
    X_test_scaled_global, y_test_global = np.array([]), np.array([])
    try:
        test_data = np.load(TEST_NPZ_FILE)
        X_test_global = test_data['embeddings']
        y_test_global = test_data['labels']
        X_test_scaled_global = scaler.transform(X_test_global)
        print(f"Loaded and scaled centralized test set '{TEST_NPZ_FILE}'. Shape: {X_test_scaled_global.shape}")
    except Exception as e:
        print(f"Warning: Could not load/scale test set '{TEST_NPZ_FILE}': {e}. Server-side evaluation might fail.")

    evaluate_fn_instance = get_evaluate_fn(X_test_scaled_global, y_test_global)

    # Import and configure FedAvgM
    from flwr.server.strategy import FedAvgM

    strategy = FedAvgM(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=EXPECTED_CLIENTS,
        min_evaluate_clients=0,
        min_available_clients=EXPECTED_CLIENTS,
        evaluate_fn=evaluate_fn_instance,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        initial_parameters=fl.common.ndarrays_to_parameters(build_model().get_weights()),
        server_learning_rate=1.0,  # default
        server_momentum=0.9 ,   # can be tuned
        on_fit_config_fn=on_fit_config_fn
    )

    config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    start_time = time.time()
    history = fl.server.start_server(server_address=SERVER_ADDRESS, config=config, strategy=strategy)
    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\n⏱️ Total Training Time: {int(minutes)} minutes, {int(seconds)} seconds")


    print("\n--- Server Finished ---")
    print("\n--- Final Run Summary (FedAvgM) ---")
    print(f"\nTotal Rounds Completed: {len(history.losses_centralized)}")

    print("\nIndividual Client Performance:")
    print(f" -> Find final 'Post-Fit Local Acc' for each client in their respective terminal outputs for the LAST round.")
    final_avg_local_metrics = history.metrics_distributed_fit.get("avg_local_accuracy")
    if final_avg_local_metrics:
        final_round_local, final_avg_local_acc = final_avg_local_metrics[-1]
        print(f"\nAverage Local Accuracy across clients (End of Round {final_round_local}): {final_avg_local_acc:.4f}")
    else:
        print("\nCould not retrieve final average local accuracy.")

    print("\nGlobal Model Performance (on Centralized Test Set):")
    if history.metrics_centralized and 'accuracy' in history.metrics_centralized:
        rounds = [r for r, _ in history.metrics_centralized['accuracy']]
        accuracies = [a for _, a in history.metrics_centralized['accuracy']]
        best_round_idx = np.argmax(accuracies)
        best_round = rounds[best_round_idx]
        best_accuracy = accuracies[best_round_idx]
        final_round, final_accuracy = history.metrics_centralized['accuracy'][-1]
        print(f" -> Final Aggregated Test Accuracy (Round {final_round}): {final_accuracy:.4f}")
        print(f" -> Best Aggregated Test Accuracy: {best_accuracy:.4f} (Achieved at Round {best_round})")
        print(f"\n   Metrics at Best Round ({best_round}):")
        for metric, values in history.metrics_centralized.items():
            value_at_best_round = next((v for r, v in values if r == best_round), None)
            if value_at_best_round is not None:
                print(f"    {metric.replace('_', ' ').title()}: {value_at_best_round:.4f}")
    else:
        print(" -> Could not retrieve final aggregated accuracy history.")
