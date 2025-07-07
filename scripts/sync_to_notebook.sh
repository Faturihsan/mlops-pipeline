#!/bin/bash

# Define your notebook pod (adjust the label if needed)
POD_NAME=$(kubectl get pod -n kubeflow -l app=jupyter -o jsonpath="{.items[0].metadata.name}")

# Exec into notebook pod and pull latest code
kubectl exec -n kubeflow "$POD_NAME" -- bash -c "
if [ ! -d /home/jovyan/mlops-project ]; then
  cd /home/jovyan && git clone https://gitlab.com/<your-user>/<your-repo>.git mlops-project
else
  cd /home/jovyan/mlops-project && git pull origin main
fi
"
