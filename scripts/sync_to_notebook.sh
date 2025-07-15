#!/bin/bash

# Define your notebook pod (adjust the label if needed)
POD_NAME=$(kubectl get pod -n kubeflow-user-example-com -o name | grep object-detection)

# Exec into notebook pod and pull latest code
kubectl exec -n kubeflow "$POD_NAME" -- bash -c "
if [ ! -d /home/jovyan/mlops-project ]; then
  cd /home/jovyan && git clone https://gitlab.com/faturihsan-skripsi/mlops-pipeline-automation.git mlops-pipeline-automation
else
  cd /home/jovyan/mlops-project && git pull origin main
fi
"
