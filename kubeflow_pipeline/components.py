import os
from kfp.components import create_component_from_func

def _download_dataset(api_key: str,
                      workspace: str,
                      project_name: str,
                      version_number: int,
                      export_format: str = "yolov8") -> str:
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    ds = version.download(export_format)
    return ds.location

download_dataset_op = create_component_from_func(
    _download_dataset,
    base_image="python:3.9",
    packages_to_install=["roboflow"]
)

def _train_model(model_name: str,
                 dataset_path: str,
                 epochs: int,
                 output_dir: str) -> str:
    from ultralytics import YOLO
    import os
    data_yaml = os.path.join(dataset_path, "data.yaml")
    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, project=output_dir, plots=True)
    return os.path.join(output_dir, "runs", "detect", "train", "weights", "best.pt")

train_model_op = create_component_from_func(
    _train_model,
    base_image="ultralytics/ultralytics:latest",  # or python:3.9 + `pip install ultralytics`
)

def _validate_model(model_path: str, dataset_path: str):
    from ultralytics import YOLO
    import os
    data_yaml = os.path.join(dataset_path, "data.yaml")
    YOLO(model_path).val(data=data_yaml)

validate_model_op = create_component_from_func(
    _validate_model,
    base_image="ultralytics/ultralytics:latest",
)

def _predict_model(model_path: str, dataset_path: str, conf: float = 0.25, save: bool = True):
    from ultralytics import YOLO
    import os
    source = os.path.join(dataset_path, "test", "images")
    YOLO(model_path).predict(source=source, conf=conf, save=save)

predict_model_op = create_component_from_func(
    _predict_model,
    base_image="ultralytics/ultralytics:latest",
)

def _export_model(model_path: str,
                  export_format: str = "onnx",
                  nms: bool = True,
                  minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local:9000",
                  minio_access_key: str = "minio",
                  minio_secret_key: str = "minio123",
                  bucket: str = "models-trained"):
    from ultralytics import YOLO
    import os
    # export
    YOLO(model_path).export(format=export_format, nms=nms)
    onnx_path = os.path.splitext(model_path)[0] + f".{export_format}"

    # upload to MinIO
    from minio import Minio
    client = Minio(minio_endpoint,
                   access_key=minio_access_key,
                   secret_key=minio_secret_key,
                   secure=False)
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    client.fput_object(bucket, os.path.basename(onnx_path), onnx_path)
    print(f"Uploaded {onnx_path} → bucket {bucket}")

export_model_op = create_component_from_func(
    _export_model,
    base_image="python:3.9",
    packages_to_install=["ultralytics", "minio"]
)
