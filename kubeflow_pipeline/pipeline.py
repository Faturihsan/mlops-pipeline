from kfp.v2.dsl import pipeline
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
    output_dir: str = "/mnt/data/output"
):
    # 1) download
    ds = download_dataset(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        version_number=version_number
    )

    # 2) train
    trained = train_model(
        model_name=model_name,
        dataset_path=ds.output,
        epochs=epochs,
        output_dir=output_dir
    )

    # 3) validate
    validate_model(
        model_path=trained.output,
        dataset_path=ds.output
    )

    # 4) predict
    predict_model(
        model_path=trained.output,
        dataset_path=ds.output
    )

    # 5) export
    export_model(
        model_path=trained.output
    )
