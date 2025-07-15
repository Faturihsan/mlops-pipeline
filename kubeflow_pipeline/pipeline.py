import os
from kfp.v2.dsl import component, pipeline
from kfp.v2 import compiler
from kfp import Client


@component
def download_dataset_component(
    api_key: str,
    workspace: str,
    project_name: str,
    version_number: int,
    export_format: str = "yolov8"
) -> str:
    """Downloads dataset via Roboflow and returns the local folder path."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(export_format)
    return dataset.location


@component
def train_model_component(
    model_name: str,
    dataset_location: str,
    epochs: int,
    project_dir: str
) -> str:
    """Trains YOLOv8 model and returns the path to best weights."""
    from ultralytics import YOLO
    data_yaml = os.path.join(dataset_location, "data.yaml")
    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, project=project_dir, plots=True)
    best_weights = os.path.join(
        project_dir, "runs", "detect", "train", "weights", "best.pt"
    )
    return best_weights


@component
def validate_model_component(
    model_path: str,
    dataset_location: str
):
    """Validates the trained model against the validation set."""
    from ultralytics import YOLO
    data_yaml = os.path.join(dataset_location, "data.yaml")
    model = YOLO(model_path)
    model.val(data=data_yaml)


@component
def predict_model_component(
    model_path: str,
    dataset_location: str,
    conf: float = 0.25,
    save: bool = True
):
    """Runs predictions on the test split and optionally saves results."""
    from ultralytics import YOLO
    source = os.path.join(dataset_location, "test", "images")
    model = YOLO(model_path)
    model.predict(source=source, conf=conf, save=save)


@component
def export_model_component(
    model_path: str,
    export_format: str = "onnx",
    nms: bool = True
):
    """Exports the trained model to the specified format (e.g. ONNX)."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.export(format=export_format, nms=nms)


@pipeline(name="yolov8-training-pipeline")
def yolov8_pipeline(
    api_key: str = "Ta6oCmhCi264c7zHQyZM",
    workspace: str = "zx-r6lu6",
    project_name: str = "student-and-non-student",
    version_number: int = 1,
    model_name: str = "yolov8s.pt",
    epochs: int = 10,
    project_dir: str = "."
):
    # 1. Download dataset
    dataset_task = download_dataset_component(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        version_number=version_number
    )

    # 2. Train model
    train_task = train_model_component(
        model_name=model_name,
        dataset_location=dataset_task.output,
        epochs=epochs,
        project_dir=project_dir
    )

    # 3. Validate model
    validate_task = validate_model_component(
        model_path=train_task.output,
        dataset_location=dataset_task.output
    )

    # 4. Predict on test set
    predict_task = predict_model_component(
        model_path=train_task.output,
        dataset_location=dataset_task.output
    )

    # 5. Export to ONNX
    export_task = export_model_component(
        model_path=train_task.output
    )


if __name__ == "__main__":
    # 1. Compile the pipeline to JSON
    compiler.Compiler().compile(
        pipeline_func=yolov8_pipeline,
        package_path="yolov8_pipeline.json"
    )

    # 2. Connect to your Kubeflow Pipelines endpoint
    client = Client(host="http://localhost:8080/pipeline")

    # 3. Upload (or update) the pipeline in your Kubeflow instance
    client.upload_pipeline(
        pipeline_package_path="yolov8_pipeline.json",
        pipeline_name="YOLOv8 Training Pipeline"
    )

    # 4. (Optional) Immediately start a run with the default args
    run = client.create_run_from_pipeline_func(
        yolov8_pipeline,
        arguments={
            "api_key": "Ta6oCmhCi264c7zHQyZM",
            "workspace": "zx-r6lu6",
            "project_name": "student-and-non-student",
            "version_number": 1,
            "model_name": "yolov8s.pt",
            "epochs": 10,
            "project_dir": "."
        }
    )
    print(f"Started run: {run.run.name} (ID: {run.run_id})")
