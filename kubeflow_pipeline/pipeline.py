import kfp
from kfp import dsl
from kfp.components import func_to_container_op

from kubeflow_pipeline.components import (
    download_dataset,
    train_model,
    validate_model,
    predict_model,
    export_model
)

# wrap functions
download_op = func_to_container_op(download_dataset, base_image="python:3.9", packages_to_install=["roboflow"])
train_op    = func_to_container_op(train_model,    base_image="python:3.9", packages_to_install=["ultralytics"])
validate_op = func_to_container_op(validate_model, base_image="python:3.9", packages_to_install=["ultralytics"])
predict_op  = func_to_container_op(predict_model,  base_image="python:3.9", packages_to_install=["ultralytics"])
export_op   = func_to_container_op(export_model,   base_image="python:3.9", packages_to_install=["ultralytics","minio"])

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
    ds = download_op(api_key, workspace, project_name, version_number)
    train = train_op(model_name, ds.output, epochs, output_dir).after(ds)
    validate_op(train.output, ds.output).after(train)
    predict_op(train.output, ds.output).after(train)
    export_op(
        train.output,
        "onnx", True,
        minio_endpoint, minio_access_key, minio_secret_key, bucket
    ).after(train)

# if __name__ == "__main__":
#     kfp.compiler.Compiler().compile(
#         pipeline_func=yolov8_pipeline,
#         package_path="yolov8_pipeline.yaml"
#     )
#     print("✅ Compiled to yolov8_pipeline.yaml")
