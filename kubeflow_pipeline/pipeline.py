import os
import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from typing import NamedTuple


# Components
def download_dataset(api_key: str, workspace: str, project_name: str, version_number: int, export_format: str = "yolov8") -> NamedTuple("Outputs", [("dataset_path", str)]):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(export_format)
    return (dataset.location,)

def train_model(model_name, dataset_path, epochs, output_dir):
    from ultralytics import YOLO
    import os
    data_yaml = os.path.join(dataset_path, "data.yaml")
    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, project=output_dir, plots=True)
    return os.path.join(output_dir, "runs", "detect", "train", "weights", "best.pt")

def validate_model(model_path, dataset_path):
    from ultralytics import YOLO
    import os
    data_yaml = os.path.join(dataset_path, "data.yaml")
    YOLO(model_path).val(data=data_yaml)

def predict_model(model_path, dataset_path, conf=0.25, save=True):
    from ultralytics import YOLO
    import os
    source = os.path.join(dataset_path, "test", "images")
    YOLO(model_path).predict(source=source, conf=conf, save=save)

def export_model(model_path, export_format="onnx", nms=True,
                 minio_endpoint="minio-service.kubeflow.svc.cluster.local:9000",
                 minio_access_key="minio", minio_secret_key="minio123", bucket="models-trained"):
    from ultralytics import YOLO
    from minio import Minio
    import os

    YOLO(model_path).export(format=export_format, nms=nms)
    base = os.path.splitext(model_path)[0]
    onnx_path = f"{base}.{export_format}"

    client = Minio(endpoint=minio_endpoint, access_key=minio_access_key,
                   secret_key=minio_secret_key, secure=False)
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    client.fput_object(bucket, os.path.basename(onnx_path), onnx_path)

# Create container components
download_op = func_to_container_op(download_dataset, base_image="python:3.9", packages_to_install=["roboflow"])
train_op    = func_to_container_op(train_model, base_image="python:3.9", packages_to_install=["ultralytics"])
validate_op = func_to_container_op(validate_model, base_image="python:3.9", packages_to_install=["ultralytics"])
predict_op  = func_to_container_op(predict_model, base_image="python:3.9", packages_to_install=["ultralytics"])
export_op   = func_to_container_op(export_model, base_image="python:3.9", packages_to_install=["ultralytics", "minio"])

@dsl.pipeline(name="yolov8-object-detection-pipeline-v1")
def yolov8_pipeline(api_key="Ta6oCmhCi264c7zHQyZM", workspace="zx-r6lu6",
                    project_name="student-and-non-student", version_number=1,
                    model_name="yolov8s.pt", epochs=10, output_dir="/mnt/data/output",
                    minio_endpoint="minio-service.kubeflow.svc.cluster.local:9000",
                    minio_access_key="minio", minio_secret_key="minio123", bucket="models-trained"):

    ds = download_op(api_key, workspace, project_name, version_number)
    tr = train_op(model_name, ds.outputs["dataset_path"], epochs, output_dir).after(ds)
    validate_op(tr.outputs["output"], ds.outputs["dataset_path"]).after(tr)
    predict_op(tr.outputs["output"], ds.outputs["dataset_path"]).after(tr)
    export_op(tr.outputs["output"], "onnx", True, minio_endpoint, minio_access_key, minio_secret_key, bucket).after(tr)


