# pipeline.py

import kfp
from kfp import dsl
from kfp.components import create_component_from_func

from components import (
    download_dataset,
    train_model,
    validate_model,
    predict_model,
    export_model
)

# wrap raw functions into ContainerOps
download_op = create_component_from_func(
    download_dataset,
    base_image="python:3.9",
    packages_to_install=["roboflow"]
)

train_op = create_component_from_func(
    train_model,
    base_image="python:3.9",
    packages_to_install=["ultralytics"]
)

validate_op = create_component_from_func(
    validate_model,
    base_image="python:3.9",
    packages_to_install=["ultralytics"]
)

predict_op = create_component_from_func(
    predict_model,
    base_image="python:3.9",
    packages_to_install=["ultralytics"]
)

export_op = create_component_from_func(
    export_model,
    base_image="python:3.9",
    packages_to_install=["ultralytics", "minio"]
)

@dsl.pipeline(
    name="yolov8-object-detection-pipeline-v1",
    description="Download → Train → Validate → Predict → Export to MinIO"
)
def yolov8_pipeline(
    api_key: str = "Ta6oCmhCi264c7zHQyZM",
    workspace: str = "zx-r6lu6",
    project_name: str = "student-and-non-student",
    version_number: int = 1,
    model_name: str = "yolov8s.pt",
    epochs: int = 10,
    output_dir: str = "/mnt/data/output",
    minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
    bucket: str = "models-trained"
):
    # 1) Download
    ds = download_op(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        version_number=version_number
    )

    # 2) Train (after Download)
    train = train_op(
        model_name=model_name,
        dataset_path=ds.output,
        epochs=epochs,
        output_dir=output_dir
    ).after(ds)

    # 3) Validate (after Train)
    validate_op(
        model_path=train.output,
        dataset_path=ds.output
    ).after(train)

    # 4) Predict (after Validate)
    predict_op(
        model_path=train.output,
        dataset_path=ds.output
    ).after(train)

    # 5) Export & push to MinIO (after Predict)
    export_op(
        model_path=train.output,
        export_format="onnx",
        nms=True,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket=bucket
    ).after(train)

# compile locally if run as script
# if __name__ == "__main__":
#     kfp.compiler.Compiler().compile(
#         pipeline_func=yolov8_pipeline,
#         package_path="yolov8_pipeline.yaml"
#     )
#     print("✅ Compiled v1 pipeline to yolov8_pipeline.yaml")
