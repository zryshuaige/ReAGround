# Optional reproducibility path for servers with full root + nvidia-container-toolkit.
# On AutoDL-style rental platforms (no docker daemon inside instances), use
# requirements.txt + scripts/setup_server.sh instead.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
WORKDIR /workspace/ReAGround
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["bash"]
