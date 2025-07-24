import kfp
from kfp.compiler import Compiler
from pipeline import yolov8_pipeline
import time

if __name__ == "__main__":
    Compiler().compile(
        pipeline_func=yolov8_pipeline,
        package_path="yolov8_pipeline.yaml"
    )
    print("✅ Compiled yolov8_pipeline.yaml")

    client = kfp.Client()
    pipeline_info = client.upload_pipeline(
        pipeline_package_path="yolov8_pipeline.yaml",
        pipeline_name="Object Detection Pipeline Test"
    )
    print(f"✅ Uploaded pipeline: id={pipeline_info.id}")

    run = client.create_run_from_pipeline_func(
        yolov8_pipeline,
        arguments={
            "api_key": "Ta6oCmhCi264c7zHQyZM",
            "workspace": "zx-r6lu6",
            "project_name": "student-and-non-student",
            "version_number": 1,
            "model_name": "yolov8s.pt",
            "epochs": 10,
            "output_dir": "/tmp/output",
            "minio_endpoint": "minio-service.kubeflow.svc.cluster.local:9000",
            "minio_access_key": "minio",
            "minio_secret_key": "minio123",
            "bucket": "models-trained",
        },
        experiment_name="YOLOv8-Experiments",
        run_name=f"auto-run-{int(time.time())}"
    )
    print("🚀 Launched run:", run.run_id)
