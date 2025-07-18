from kfp.dsl import pipeline
from components import (
    download_dataset,
    train_model,
    validate_model,
    predict_model,
    export_model
)

@pipeline(name="yolov8-object-detection-pipeline")
def yolov8_pipeline(
    api_key: str = "Ta6oCmhCi264c7zHQyZM",
    workspace: str = "zx-r6lu6",
    project_name: str = "student-and-non-student",
    version_number: int = 1,
    model_name: str = "yolov8s.pt",
    epochs: int = 10,
    output_dir: str = "/mnt/data/output",

    # MinIO parameters
    minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
    bucket: str = "models-trained"
):
    ds = download_dataset(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        version_number=version_number
    )

    trained = train_model(
        model_name=model_name,
        dataset_path=ds.output,
        epochs=epochs,
        output_dir=output_dir
    )

    validate_model(
        model_path=trained.output,
        dataset_path=ds.output
    )

    predict_model(
        model_path=trained.output,
        dataset_path=ds.output
    )

    export_model(
        model_path=trained.output,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket=bucket
    )
