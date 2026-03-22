FROM continuumio/miniconda3:latest

WORKDIR /framework

ENV MPLBACKEND=Agg
ENV PYTHONPATH=/framework:/framework/xai_framework

COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean -afy && \
    find /opt/conda/envs/xai_env -name "*.pyc" -delete && \
    find /opt/conda/envs/xai_env -name "__pycache__" -type d -exec rm -rf {} +

SHELL ["conda", "run", "-n", "xai_env", "/bin/bash", "-c"]

COPY xai_framework/ ./xai_framework/
COPY scripts/ ./scripts/
COPY config/ ./config/

CMD ["conda", "run", "--no-capture-output", "-n", "xai_env", "tail", "-f", "/dev/null"]
