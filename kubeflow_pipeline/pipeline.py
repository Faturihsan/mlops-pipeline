import os
import glob
from ultralytics import YOLO
from roboflow import Roboflow


def download_dataset(api_key: str,
                     workspace: str,
                     project_name: str,
                     version_number: int,
                     export_format: str = "yolov8"):
    """
    Download dataset dari Roboflow.
    Mengembalikan objek dataset dengan atribut `location`.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(export_format)
    return dataset


def train_model(model_name: str = "yolov8s.pt",
                data_yaml: str = None,
                epochs: int = 10,
                project_dir: str = ".") -> str:
    """
    Train model YOLOv8.
    Mengembalikan path ke berkas best.pt setelah training.
    """
    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, project=project_dir, plots=True)
    best_weights = os.path.join(
        project_dir, "runs", "detect", "train", "weights", "best.pt")
    return best_weights


def validate_model(model_path: str,
                   data_yaml: str):
    """
    Validasi model YOLOv8 yang sudah di-train.
    """
    model = YOLO(model_path)
    model.val(data=data_yaml)


def predict_model(model_path: str,
                  source: str,
                  conf: float = 0.25,
                  save: bool = True):
    """
    Jalankan prediksi pada dataset.
    """
    model = YOLO(model_path)
    results = model.predict(source=source, conf=conf, save=save)
    return results


def export_model(model_path: str,
                 export_format: str = "onnx",
                 nms: bool = True):
    """
    Export model yang sudah di-train ke format tertentu (misal ONNX).
    """
    model = YOLO(model_path)
    model.export(format=export_format, nms=nms)


def main():
    # Set direktori kerja
    HOME = os.getcwd()

    # Parameter untuk download dataset
    API_KEY = "Ta6oCmhCi264c7zHQyZM"
    WORKSPACE= "zx-r6lu6"
    PROJECT_NAME= "student-and-non-student"
    VERSION_NUMBER= 1

    # Download dataset
    dataset= download_dataset(
        API_KEY, WORKSPACE, PROJECT_NAME, VERSION_NUMBER)
    data_yaml= os.path.join(dataset.location, "data.yaml")

    # Train model
    best_model_path= train_model(
        model_name = "yolov8s.pt",
        data_yaml = data_yaml,
        epochs = 10,
        project_dir = HOME
    )

    # Validasi model
    validate_model(best_model_path, data_yaml)

    # Prediksi pada test set
    test_images= os.path.join(dataset.location, "test", "images")
    predict_model(best_model_path, source=test_images, conf=0.25, save=True)

    # Export model ke ONNX
    export_model(best_model_path, export_format="onnx", nms=True)


if __name__ == "__main__":
    main()
