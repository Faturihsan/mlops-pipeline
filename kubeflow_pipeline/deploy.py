from kfp.v2 import compiler
from kfp import Client
from pipeline import yolov8_pipeline

if __name__ == "__main__":
    # 1) Compile to JSON
    compiler.Compiler().compile(
        pipeline_func=yolov8_pipeline,
        package_path="yolov8_pipeline.json"
    )
    print("✅ Compiled yolov8_pipeline.json")

    # 2) Upload to Kubeflow (ensure kubectl port-forward is active)
    client = Client(host="http://localhost:8085/pipeline")
    client.upload_pipeline(
        pipeline_package_path="yolov8_pipeline.json",
        pipeline_name="YOLOv8 Object Detection"
    )
    print("✅ Uploaded pipeline to Kubeflow")
