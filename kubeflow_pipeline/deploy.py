import kfp
from kfp.v2 import compiler
from kfp import Client
from pipeline import yolov8_pipeline

if __name__ == "__main__":
    # Compile the pipeline to JSON
    compiler.Compiler().compile(
        pipeline_func=yolov8_pipeline,
        package_path="yolov8_pipeline.json"
    )
    print("✅ Compiled yolov8_pipeline.json")   

    # Upload to Kubeflow Pipelines (requires port-forward svc/ml-pipeline)
    client = kfp.Client()
    client.upload_pipeline(
        pipeline_package_path="yolov8_pipeline.json",
        pipeline_name="Object Detection TestV1"
    )
    print("✅ Uploaded pipeline to Kubeflow")
import kfp
from kfp.v2 import compiler
from pipeline import yolov8_pipeline

if __name__ == "__main__":
    # 1) Compile
    compiler.Compiler().compile(
        pipeline_func=yolov8_pipeline,
        package_path="yolov8_pipeline.json"
    )
    print("✅ Compiled yolov8_pipeline.json")

    # 2) Upload (in-cluster or via port-forward svc/ml-pipeline)
    client = kfp.Client()
    client.upload_pipeline(
        pipeline_package_path="yolov8_pipeline.json",
        pipeline_name="Object Detection Test_1"
    )
    print("✅ Uploaded pipeline to Kubeflow")

    # 3) Immediately run it with your default or custom parameters
    run = client.create_run_from_pipeline_func(
        yolov8_pipeline,
        arguments={
            "api_key":       "Ta6oCmhCi264c7zHQyZM",
            "workspace":     "zx-r6lu6",
            "project_name":  "student-and-non-student",
            "version_number": 1,
            "model_name":    "yolov8s.pt",
            "epochs":        10,
            "output_dir":    "/mnt/data/output",
            # add minio params if your pipeline needs them:
            "minio_endpoint": "minio-service.kubeflow.svc.cluster.local:9000",
            "minio_access_key": "minio",
            "minio_secret_key": "minio123",
            "bucket": "models-trained",
        },
        experiment_name="YOLOv8-Experiments",
        run_name="auto-run-"+kfp.dsl.PIPELINE_JOB_ID_PLACEHOLDER
    )
    print(f"🚀 Launched run: {run.run_id}")
