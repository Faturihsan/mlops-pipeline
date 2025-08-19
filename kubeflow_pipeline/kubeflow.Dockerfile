
# kubeflow.Dockerfile

# Mulai dari base image yang sama dengan yang Anda gunakan untuk training
FROM jupyter/scipy-notebook:python-3.9

# Salin file requirements Anda ke dalam image
COPY requirements.txt .

# Install semua package dari requirements.txt
# Menambahkan --timeout untuk memberi waktu lebih saat mengunduh
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --timeout=600