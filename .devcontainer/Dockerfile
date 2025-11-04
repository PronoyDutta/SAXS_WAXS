FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.11

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# Install Python packages (quotes around numpy constraint)
RUN pip install --no-cache-dir "numpy<2" matplotlib pandas scipy ipywidgets pybaselines galvani eclabfiles \
   && pip install --no-cache-dir -r /tmp/requirements.txt

USER vscode
WORKDIR /workspace
