ARG JLAB_TAG="latest"
ARG PROXY=""
FROM jupyter/scipy-notebook:${JLAB_TAG}

WORKDIR /home/jovyan/work
ENV http_proxy PROXY
ENV https_proxy PROXY

COPY ./pyproject.toml ./poetry.lock* /home/jovyan/work/
COPY . /home/jovyan/work/
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install && \
    rm -rf ~/.cache/pypoetry/{cache,artifacts} && \
    pip cache purge
