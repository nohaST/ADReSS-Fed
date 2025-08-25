# ADReSS-Fed
A federated learning project for privacy-preserving Alzheimer's disease detection using speech data from the ADReSS dataset. 

This repository contains the code for a federated learning project focused on privacy-preserving early-stage Alzheimer's disease detection using speech data from the ADReSS dataset.

## Key Features
Federated Learning Implementation: Utilizes the Flower framework to train models across decentralized clients without sharing raw data.

Comparative Analysis: Includes experiments comparing different components of the FL pipeline, such as:

Classification models (e.g., lightweight CNNs, MLPs).

Aggregation techniques.

The impact of varying the number of clients.

Audio Feature Extraction: The project uses VGGish embeddings as the input features, which are extracted from the raw audio files.

Reproducibility: The code is designed to be easily reproducible, with clear configurations for hyperparameters and data splits.
