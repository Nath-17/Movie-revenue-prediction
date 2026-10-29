import mlflow
from mlflow.tracking import MlflowClient


def register_model_with_alias(model, registry_name: str, alias: str, artifact_path: str = "model"):
    # Registration should not fight the active parent run; it should become
    # a nested child when an optimization run is active.
    nested = mlflow.active_run() is not None

    with mlflow.start_run(run_name=alias, nested=nested):
        mlflow.xgboost.log_model(
            xgb_model=model,
            name=artifact_path,
            registered_model_name=registry_name,
        )

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{registry_name}'")
    latest_version = max(int(v.version) for v in versions)
    client.set_registered_model_alias(name=registry_name, alias=alias, version=latest_version)

def load_model_by_alias(registry_name: str, alias: str):
    return mlflow.pyfunc.load_model(f"models:/{registry_name}@{alias}")