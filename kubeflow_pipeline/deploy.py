import kfp
from kubeflow_pipeline.pipeline import yolov8_pipeline

if __name__ == "__main__":
    # 1) compile
    kfp.compiler.Compiler().compile(
        pipeline_func=yolov8_pipeline,
        package_path="yolov8_pipeline.json"
    )
    print("✅ Compiled yolov8_pipeline.json")

    # 2) upload
    client = kfp.Client()  # in-cluster or via port-forward
    client.upload_pipeline(
        pipeline_package_path="yolov8_pipeline.json",
        pipeline_name="Object Detection Test v1"
    )
    print("✅ Uploaded pipeline")

    # 3) immediate run
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
            "minio_endpoint": "minio-service.kubeflow.svc.cluster.local:9000",
            "minio_access_key":"minio",
            "minio_secret_key":"minio123",
            "bucket":        "models-trained",
        },
        experiment_name="YOLOv8-Experiments",
        run_name="auto-"+kfp.dsl.PIPELINE_JOB_ID_PLACEHOLDER
    )
    print("🚀 Launched:", run.run_id)
