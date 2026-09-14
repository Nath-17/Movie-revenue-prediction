# ML Challenge — Movie Revenue Prediction

This project was forked from a machine learning challenge carried out during my Master's degree. The goal was to develop a notebook for data processing and model training, then run it on an unknown test set whose true labels were only available through the course platform.

## Project objective

This project focuses on building a cleaner code structure, introducing MLflow tracking, improving model optimization, and creating a small Streamlit interface to present the results.

The repository is organized around a simple structure:

- data loading and preprocessing in the data package
- model configuration and training logic in the model package
- configuration files for paths, preprocessing rules, and modeling options
- a prototype Streamlit entry point at the project root



## Repository structure

- `src/data/` contains the data loader and preprocessing logic
- `src/models/` contains the modeling and training configuration
- `config/` contains YAML files for paths, preprocessing, and model parameters
- `data/inputs/` stores local data copied from the challenge sources
- `streamlit_app.py` is the intended entry point for a user-facing interface

## Local installation

Requirements:

- Python 3.10 or newer
- pip or uv

Clone the repository:

```bash
git clone <repository-url>
cd Movie-revenue-prediction
```

Create a virtual environment and install the project dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```
Data are loaded from the local folder `data/inputs/` when it exists. If the folder is missing, the loader can fetch the challenge files from the configured online source.

## Run the data pipeline

The project currently exposes a Python workflow for loading and preprocessing the challenge data:

```bash
python src/scripts/workflow_train.py
```

## What remains to do

This repository is still a working project and not a finished product. The main next steps are:

- connect an MLflow registry
- load tuned models
- finish the Streamlit interface in `streamlit_app.py`
- containerize the code

## Current status

The project already contains:

- a preprocessing pipeline for movie metadata
- model configuration files


