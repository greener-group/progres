#!/bin/bash

set -eu

source activate ${CONDA_ENV_NAME}

$@
