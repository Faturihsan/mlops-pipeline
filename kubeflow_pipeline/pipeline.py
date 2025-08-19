# kubeflow_pipeline/pipeline.py

import os
from typing import NamedTuple

import kfp
from kfp import dsl
from kfp.components import func_to_container_op


# ----------------------
# Component functions
# ----------------------

# def read_requirements(file_path="requirements.txt"):
#     """Membaca file requirements.txt dan mengembalikannya sebagai list."""
#     with open(file_path, 'r') as f:
#         return [line.strip() for line in f if line.strip()]
    

def download_dataset(api_key: str,
                     workspace: str,
                     project_name: str,
                     version_number: int,
                     export_format: str = "yolov8",
                     base_dir: str = "/mnt/work/rf") -> NamedTuple("Outputs", [("dataset_path", str)]):
    """
    Downloads a dataset from Roboflow into a shared volume directory (base_dir)
    and returns the absolute dataset folder path.
    """
    import os
    os.makedirs(base_dir, exist_ok=True)
    os.chdir(base_dir)  # ensure Roboflow writes inside the shared PVC

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(export_format)

    # dataset.location is the absolute folder where data.yaml resides
    return (dataset.location,)


def train_model(model_name: str,
                dataset_path: str,
                epochs: int,
                output_dir: str = "/mnt/work/output") -> NamedTuple("Outputs", [("model_path", str)]):
    """
    Trains YOLOv8 using the dataset at dataset_path and writes runs to output_dir.
    Both must live on the shared PVC.
    """
    import os
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)

    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"{data_yaml} not found. Check that download step wrote to the shared volume.")

    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, project=output_dir, plots=True)

    best = os.path.join(output_dir, "train", "weights", "best.pt")
    # Ultralytics default save_dir is <project>/<name>, here name defaults to 'train'
    # If your run directory is different, adjust accordingly.
    if not os.path.exists(best):
        # fallback to the other common layout
        best = os.path.join(output_dir, "runs", "detect", "train", "weights", "best.pt")
    return (best,)


def validate_model(model_path: str, dataset_path: str):
    import os
    from ultralytics import YOLO

    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"{data_yaml} not found for validation.")
    YOLO(model_path).val(data=data_yaml)


def predict_model(model_path: str, dataset_path: str, conf: float = 0.25, save: bool = True):
    import os
    from ultralytics import YOLO

    source = os.path.join(dataset_path, "test", "images")
    YOLO(model_path).predict(source=source, conf=conf, save=save)


def export_model(model_path: str,
                 export_format: str = "onnx",
                 nms: bool = True,
                 minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local:9000",
                 minio_access_key: str = "minio",
                 minio_secret_key: str = "minio123",
                 bucket: str = "models-trained"):
    """
    Exports the trained model to ONNX (on the shared volume), then uploads to MinIO.
    """
    import os
    from ultralytics import YOLO
    from minio import Minio

    YOLO(model_path).export(format=export_format, nms=nms)
    base = os.path.splitext(model_path)[0]
    onnx_path = f"{base}.{export_format}"
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Export failed: {onnx_path} not found.")

    client = Minio(endpoint=minio_endpoint, access_key=minio_access_key, secret_key=minio_secret_key, secure=False)
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    client.fput_object(bucket, os.path.basename(onnx_path), onnx_path)


# ----------------------
# Wrap functions to container components
# ----------------------
# requirements_list = read_requirements()

download_op = func_to_container_op(
    download_dataset,
    base_image="python:3.9",
    # packages_to_install=requirements_list
)

# Use an image with system libs (libGL, etc.) to avoid OpenCV issues
# All subsequent steps now read dependencies from your requirements.txt file
train_op = func_to_container_op(
    train_model,
    base_image="faturihsan/requirements:latest",
)

validate_op = func_to_container_op(z
    validate_model,
    base_image="faturihsan/requirements:latest",
)

predict_op = func_to_container_op(
    predict_model,
    base_image="faturihsan/requirements:latest",
)

export_op = func_to_container_op(
    export_model,
    base_image="faturihsan/requirements:latest",
)


@dsl.pipeline(
    name="yolov8-object-detection-pipeline-v1",
    description="Download → Train → Validate → Predict → Export to MinIO (with shared PVC)"
)
def yolov8_pipeline(api_key: str = "Ta6oCmhCi264c7zHQyZM",
                    workspace: str = "zx-r6lu6",
                    project_name: str = "student-and-non-student",
                    version_number: int = 1,
                    model_name: str = "yolov8s.pt",
                    epochs: int = 10,
                    # everything below lives on the shared volume
                    mount_path: str = "/mnt/work",
                    output_dir: str = "/mnt/work/output",
                    # MinIO
                    minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local:9000",
                    minio_access_key: str = "minio",
                    minio_secret_key: str = "minio123",
                    bucket: str = "models-trained"):

    # # 0) Create a PVC and mount it to every step
    # vol = dsl.VolumeOp(
    #     name="create-shared-volume",
    #     resource_name="yolov8-shared-pvc",
    #     size="20Gi",
    #     modes=["ReadWriteOnce"],
    # )

    pvc_name = "yolov8-shared-pvc"
    vol = dsl.PipelineVolume(pvc=pvc_name)

    # 1) Download (no cache; write into the PVC)
    ds = download_op(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        version_number=version_number,
        base_dir=f"{mount_path}/rf"
    ).set_caching_options(enable_caching=False)
    # ds.after(vol)
    ds.add_pvolumes({mount_path: vol})

    # 2) Train
    tr = train_op(
        model_name=model_name,
        dataset_path=ds.outputs["dataset_path"],
        epochs=epochs,
        output_dir=output_dir
    ).after(ds)
    tr.add_pvolumes({mount_path: vol})

    # 3) Validate
    va = validate_op(
        model_path=tr.outputs["model_path"],
        dataset_path=ds.outputs["dataset_path"]
    ).after(tr)
    va.add_pvolumes({mount_path: vol})

    # 4) Predict
    pr = predict_op(
        model_path=tr.outputs["model_path"],
        dataset_path=ds.outputs["dataset_path"]
    ).after(tr)
    pr.add_pvolumes({mount_path: vol})

    # 5) Export to MinIO
    ex = export_op(
        model_path=tr.outputs["model_path"],
        export_format="onnx",
        nms=True,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket=bucket
    ).after(tr)
    ex.add_pvolumes({mount_path: vol})