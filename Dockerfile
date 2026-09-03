# #26: CPU-only image for the geometry pipeline (tracking, events, fusion,
# demo/eval tooling) - vlm.backend=none, reid.backend=none, everything that
# needs a GPU stays on Kaggle per this repo's own architecture (see
# CLAUDE.md: "No local GPU. Never pip install, train, or run inference
# here"). This image lets the non-GPU parts run somewhere reproducible; it is
# NOT a deployment target for the VLM judge or training - there is no CUDA
# base image, no nvidia-container-toolkit assumption, and no attempt to match
# Kaggle's torch build. Unverified: built from the same static review as the
# rest of this second pass, never actually run.
FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0: opencv-python's runtime deps on a slim base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only subset: skip the GPU-only / heavy rows. Uncomment more if you
# extend this image to run the VLM stage in a CUDA-based image instead.
RUN pip install --no-cache-dir \
    ultralytics==8.3.40 opencv-python-headless==4.10.0.84 numpy==1.26.4 \
    scikit-learn==1.5.2 pyyaml==6.0.2 pillow==10.4.0

COPY . .

RUN mkdir -p out

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
