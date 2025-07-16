import os
from kfp.v2.dsl import component

@component
def download_dataset(
    api_key: str,
    workspace: str,
    project_name: str,
    version_number: int,
    export_format: str = "yolov8"
) -> str:
    """Downloads dataset via Roboflow and returns local folder path."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project_name).version(version_number)
    folder = ds.download(export_format)
    return folder.location

@component
def train_model(
    model_name: str,
    dataset_path: str,
    epochs: int,
    output_dir: str
) -> str:
    """Trains YOLOv8 and returns path to best.pt."""
    from ultralytics import YOLO
    yaml = os.path.join(dataset_path, "data.yaml")
    model = YOLO(model_name)
    model.train(data=yaml, epochs=epochs, project=output_dir, plots=True)
    return os.path.join(output_dir, "runs", "detect", "train", "weights", "best.pt")

@component
def validate_model(
    model_path: str,
    dataset_path: str
):
    """Validates model on the dataset’s validation split."""
    from ultralytics import YOLO
    yaml = os.path.join(dataset_path, "data.yaml")
    YOLO(model_path).val(data=yaml)

@component
def predict_model(
    model_path: str,
    dataset_path: str,
    conf: float = 0.25,
    save: bool = True
):
    """Runs inference on test images."""
    from ultralytics import YOLO
    source = os.path.join(dataset_path, "test", "images")
    YOLO(model_path).predict(source=source, conf=conf, save=save)

@component
def export_model(
    model_path: str,
    export_format: str = "onnx",
    nms: bool = True
):
    """Exports the trained model to ONNX (or other formats)."""
    from ultralytics import YOLO
    YOLO(model_path).export(format=export_format, nms=nms)
