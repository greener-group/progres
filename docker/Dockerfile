# can build dockerfile with --build-arg PERSISTENCE_DIR="/persist/progres" and run it with -v /host/path/to/progres/data:/persist/progres
# so that the files can be downloaded once but used many times. In particular useful in an HPC context, for the default dbs that can be pre-downloaded
# to a shared read-only directory

ARG MINICONDA_VERSION="Miniconda3-py311_24.1.2-0-Linux-x86_64"
ARG CONDA_INSTALL_PATH="/pub/conda"
ARG CONDA_ENV_NAME="progres_env"
ARG PERSISTENCE_DIR="/persist/progres"

#progres is the official repo on pypi, can override with an alternative for testing
# e.g. git+https://github.com/xeniorn/progres.git@main
ARG PROGRES_REPO='progres' 

FROM debian:12 as build

ARG CONDA_INSTALL_PATH
ARG MINICONDA_VERSION

RUN apt-get -y update \
    && apt-get -y install wget git \
    && apt-get -y autoclean

RUN wget --quiet \
    https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh \
    && bash ${MINICONDA_VERSION}.sh -bfp ${CONDA_INSTALL_PATH} \
    && rm -f ${MINICONDA_VERSION}.sh

ENV PATH="${CONDA_INSTALL_PATH}/bin:${PATH}"

ARG CONDA_ENV_NAME
ENV CONDA_ENV_NAME=${CONDA_ENV_NAME}

RUN . ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh \
    && conda create --yes -n ${CONDA_ENV_NAME} python=3.9 \
    && conda activate ${CONDA_ENV_NAME} \
    && conda install --yes pytorch=1.11 faiss-cpu -c pytorch \
    && conda install --yes pytorch-scatter pyg -c pyg \
    && conda clean --yes --all

ARG PROGRES_REPO

RUN . ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh \
    && conda activate ${CONDA_ENV_NAME} \
    && yes | pip install "${PROGRES_REPO}" \
    && pip cache purge

ARG PERSISTENCE_DIR
ENV PROGRES_DATA_DIR=${PERSISTENCE_DIR}

ARG RUN_SCRIPT="/pub/run.sh"
ENV PROGRES_RUN_SCRIPT=${RUN_SCRIPT}

COPY "docker/docker_runner.sh" ${RUN_SCRIPT}
RUN chmod +rx ${RUN_SCRIPT}

ENTRYPOINT [ "/pub/run.sh" ]


