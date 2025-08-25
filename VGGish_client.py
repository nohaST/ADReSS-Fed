import flwr as fl
import tensorflow as tf
import numpy as np
import joblib # To load the scaler
import os
import random
import argparse # To get client ID and server address from command line
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping

# C:\Users\nohas\OneDrive\Desktop\Project\
# --- Configuration ---
DEFAULT_SERVER_ADDRESS = "127.0.0.1:8080"
CLIENT_TRAIN_PATH_TEMPLATE = "./clients/vggish_3C_train_client{}.npz"
CLIENT_VAL_PATH_TEMPLATE = "./clients/vggish_3C_val_client{}.npz"
CLIENT_TEST_PATH_TEMPLATE = "./clients/vggish_3C_test_client{}.npz"
SCALER_FILE = 'global_scaler.pkl'

# Model Hyperparameters
INPUT_DIM = 128
UNITS1 = 128
UNITS2 = 64
UNITS3 = 32
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

# Training Config
LOCAL_EPOCHS = 100 
BATCH_SIZE = 32
LOCAL_ES_PATIENCE = 15 

# --- Seeding for Reproducibility ---
SEED = 42
print(f"--- Setting Random Seed: {SEED} ---")
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED) # Also sets TF seed and enables some reproducibility options


# --- Model Definition ---
def build_model(input_shape_dim=INPUT_DIM):
    # (Model definition remains the same as before)
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Flower Client Definition ---
class FlowerClient(fl.client.NumPyClient):
    # (Keep __init__, get_parameters, set_parameters as before)
    def __init__(self, client_id, model, x_train, y_train, x_val, y_val, x_test, y_test):
        self.client_id = client_id
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        batch_size = int(config.get("batch_size", BATCH_SIZE))
        server_round = int(config.get("server_round", 0))

        early_stopper = EarlyStopping(
            monitor='val_loss', patience=LOCAL_ES_PATIENCE, verbose=0, restore_best_weights=True
        )

        # Train the model - ensure shuffle=False
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            callbacks=[early_stopper],
            verbose=0
        )
        actual_epochs = len(history.history['loss'])

        # Evaluate locally AFTER training using the SEPARATE LOCAL TEST SET
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        #print ("Noha", config)
        print(f"  [Client {self.client_id} R{server_round:>2}] Fit completed ({actual_epochs} epochs). Post-Fit Local Acc (on local test set): {accuracy:.4f}, Loss: {loss:.4f}")
        updated_parameters = self.get_parameters(config={})
        return updated_parameters, len(self.x_train), {"local_accuracy": accuracy, "local_loss": loss, "local_epochs_run": actual_epochs}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

# --- Main Client Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client-id", type=str, required=True, help="Client ID (e.g., 0 or 1)")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER_ADDRESS, help=f"Server address (default: {DEFAULT_SERVER_ADDRESS})")
    args = parser.parse_args()

    # *** Added Logging ***
    print(f"\n--- Initializing Client ID: {args.client_id} ---")
    print(f"Attempting to connect to server: {args.server}")

    # --- Load Global Scaler ---
    print(f"Loading global scaler from: {SCALER_FILE}")
    try: scaler = joblib.load(SCALER_FILE)
    except Exception as e: print(f"Error loading scaler '{SCALER_FILE}': {e}"); exit()
    print(f" Global scaler loaded successfully.")

    # --- Load Client-Specific TRAINING Data ---
    client_train_data_path = CLIENT_TRAIN_PATH_TEMPLATE.format(args.client_id)
    print(f"Loading TRAIN data from: {client_train_data_path}")
    try:
        train_data = np.load(client_train_data_path); x_train_local = train_data['embeddings']; y_train_local = train_data['labels']
    except Exception as e: print(f"ERROR: Failed to load TRAIN data: {e}"); exit()
    print(f" Loaded TRAIN data. Shape X: {x_train_local.shape}, y: {y_train_local.shape}")

    # --- Load Client-Specific VALIDATION Data ---
    client_val_data_path = CLIENT_VAL_PATH_TEMPLATE.format(args.client_id)
    print(f"Loading VALIDATION data from: {client_val_data_path}")
    try:
        val_data = np.load(client_val_data_path); x_val_local = val_data['embeddings']; y_val_local = val_data['labels']
    except Exception as e: print(f"ERROR: Failed to load VALIDATION data: {e}"); exit()
    print(f" Loaded VALIDATION data. Shape X: {x_val_local.shape}, y: {y_val_local.shape}")

    # --- Load Client-Specific TESTING Data ---
    client_test_data_path = CLIENT_TEST_PATH_TEMPLATE.format(args.client_id)
    print(f"Loading TEST data from: {client_test_data_path}")
    try:
        test_data = np.load(client_test_data_path); x_test_local = test_data['embeddings']; y_test_local = test_data['labels']
    except Exception as e: print(f"ERROR: Failed to load TEST data: {e}"); exit()
    print(f" Loaded TEST data. Shape X: {x_test_local.shape}, y: {y_test_local.shape}")

    # --- Scale Data ---
    print("Scaling loaded data using global scaler...")
    try:
        x_train_scaled = scaler.transform(x_train_local)
        x_val_scaled = scaler.transform(x_val_local)
        x_test_scaled = scaler.transform(x_test_local)
    except Exception as e:
        print(f"ERROR: Failed to scale data: {e}")
        exit()
    print(f" Data scaled. Train shape: {x_train_scaled.shape}, Val shape: {x_val_scaled.shape}, Test shape: {x_test_scaled.shape}")

    # --- Create Model ---
    # Model building happens once per client script run
    print("Building Keras model...")
    model = build_model()
    print("Model built.")

    # --- Instantiate Client ---
    client = FlowerClient(
        client_id=args.client_id, model=model,
        x_train=x_train_scaled, y_train=y_train_local,
        x_val=x_val_scaled,     y_val=y_val_local,
        x_test=x_test_scaled,   y_test=y_test_local
    )

    # --- Start Client ---
    fl_error = None
    try:
        print(f"\nClient {args.client_id}: Connecting to server and starting FL rounds...")
        fl.client.start_numpy_client(server_address=args.server, client=client)
        print(f"\nClient {args.client_id}: FL rounds completed (disconnected from server).")
    except Exception as e:
         fl_error = e
         print(f"ERROR: Client {args.client_id} connection or execution failed: {e}")

    # --- Final Local Evaluation on TEST set ---
    print(f"\n--- Client {args.client_id}: Final Local Evaluation ---")
    try:
        final_loss, final_accuracy = client.model.evaluate(client.x_test, client.y_test, verbose=0)
        print(f" Final Local Model Accuracy on local test set: {final_accuracy:.4f}")
        print(f" Final Local Model Loss on local test set: {final_loss:.4f}")
    except Exception as eval_e: print(f" Error during final local evaluation: {eval_e}")

    print(f"\nClient {args.client_id} finished.")
    if fl_error: pass