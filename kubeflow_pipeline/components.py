# components.py

import os
from roboflow import Roboflow
from ultralytics import YOLO
from minio import Minio

def download_dataset(
    api_key: str,
    workspace: str,
    project_name: str,
    version_number: int,
    export_format: str = "yolov8"
) -> str:
    """
    Download dataset dari Roboflow.
    Returns the local folder path.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(export_format)
    return dataset.location

def train_model(
    model_name: str,
    dataset_path: str,
    epochs: int,
    output_dir: str
) -> str:
    """
    Train model YOLOv8.
    Returns the path to best.pt after training.
    """
    data_yaml = os.path.join(dataset_path, "data.yaml")
    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, project=output_dir, plots=True)
    return os.path.join(output_dir, "runs", "detect", "train", "weights", "best.pt")

def validate_model(
    model_path: str,
    dataset_path: str
):
    """
    Validasi model YOLOv8 yang sudah di-train.
    """
    data_yaml = os.path.join(dataset_path, "data.yaml")
    YOLO(model_path).val(data=data_yaml)

def predict_model(
    model_path: str,
    dataset_path: str,
    conf: float = 0.25,
    save: bool = True
):
    """
    Jalankan prediksi pada dataset test/images.
    """
    source = os.path.join(dataset_path, "test", "images")
    YOLO(model_path).predict(source=source, conf=conf, save=save)

def export_model(
    model_path: str,
    export_format: str = "onnx",
    nms: bool = True,
    minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
    bucket: str = "models-trained"
):
    """
    Exports the trained model to ONNX, then uploads that ONNX file to MinIO.
    """
    # 1) export locally
    YOLO(model_path).export(format=export_format, nms=nms)

    # 2) build ONNX path
    base = os.path.splitext(model_path)[0]
    onnx_path = f"{base}.{export_format}"

    # 3) push to MinIO
    client = Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    object_name = os.path.basename(onnx_path)
    client.fput_object(bucket, object_name, onnx_path)
    print(f"✅ Uploaded {object_name} to MinIO bucket {bucket}")
