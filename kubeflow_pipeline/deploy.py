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
        pipeline_name="YOLOv8 Object Detection"
    )
    print("✅ Uploaded pipeline to Kubeflow")
