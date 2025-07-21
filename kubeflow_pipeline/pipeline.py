from kfp import dsl
from kubeflow_pipeline.components import (
    download_dataset_op,
    train_model_op,
    validate_model_op,
    predict_model_op,
    export_model_op,
)

@dsl.pipeline(
    name="yolov8-object-detection-pipeline",
    description="Download → Train → Validate → Predict → Export→MinIO"
)
def yolov8_pipeline(
    api_key: str = "Ta6oCmhCi264c7zHQyZM",
    workspace: str = "zx-r6lu6",
    project_name: str = "student-and-non-student",
    version_number: int = 1,
    model_name: str = "yolov8s.pt",
    epochs: int = 10,
    output_dir: str = "/mnt/data/output",
    # MinIO params (if you want override defaults)
    minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
    bucket: str = "models-trained"
):
    ds = download_dataset_op(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        version_number=version_number
    )

    tr = train_model_op(
        model_name=model_name,
        dataset_path=ds.output,
        epochs=epochs,
        output_dir=output_dir
    )

    validate_model_op(
        model_path=tr.output,
        dataset_path=ds.output
    )

    predict_model_op(
        model_path=tr.output,
        dataset_path=ds.output
    )

    export_model_op(
        model_path=tr.output,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket=bucket
    )
