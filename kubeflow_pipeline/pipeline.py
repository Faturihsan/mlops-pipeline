from kfp.v2 import compiler
from kfp import Client
from kfp.v2.dsl import component, pipeline


@component
def download_dataset_component(api_key: str, workspace: str, project_name: str, version_number: int, export_format: str = "yolov8") -> str:
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(export_format)
    return dataset.location


@component
def train_model_component(model_name: str, data_yaml: str, epochs: int, project_dir: str) -> str:
    from ultralytics import YOLO
    import os
    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, project=project_dir, plots=True)
    best_weights = os.path.join(project_dir, "runs", "detect", "train", "weights", "best.pt")
    return best_weights


@component
def validate_model_component(model_path: str, data_yaml: str):
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.val(data=data_yaml)


@component
def predict_model_component(model_path: str, source: str, conf: float = 0.25, save: bool = True):
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.predict(source=source, conf=conf, save=save)


@component
def export_model_component(model_path: str, export_format: str = "onnx", nms: bool = True):
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.export(format=export_format, nms=nms)


@pipeline(name="yolov8-training-pipeline")
def yolov8_pipeline(
    api_key: str = "Ta6oCmhCi264c7zHQyZM",
    workspace: str = "zx-r6lu6",
    project_name: str = "student-and-non-student",
    version_number: int = 1,
    model_name: str = "yolov8s.pt",
    epochs: int = 10,
    project_dir: str = "."
):
    dataset = download_dataset_component(api_key, workspace, project_name, version_number)
    train = train_model_component(model_name, data_yaml=dataset.output + "/data.yaml", epochs=epochs, project_dir=project_dir)
    validate = validate_model_component(train.output, data_yaml=dataset.output + "/data.yaml")
    predict = predict_model_component(train.output, source=dataset.output + "/test/images")
    export = export_model_component(train.output)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=yolov8_pipeline,
        package_path="yolov8_pipeline.json"
    )

    client = Client(host="http://localhost:8080/pipeline")
    client.upload_pipeline(
        pipeline_package_path="yolov8_pipeline.json",
        pipeline_name="YOLOv8 Training Pipeline"
    )
