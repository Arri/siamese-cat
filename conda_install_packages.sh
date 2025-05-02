#!/bin/bash

# Create a new conda environment (optional)
ENV_NAME="cats"


# Check if we are already inside the conda environment
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    echo "Already inside the conda environment '$ENV_NAME'. Skipping activation."
else
    # Check if the environment exists
    if conda info --envs | grep -q "^$ENV_NAME "; then
        echo "Conda environment '$ENV_NAME' already exists. Activating..."
    else
        echo "Creating conda environment '$ENV_NAME'..."
        conda create -y -n $ENV_NAME python=3.9  # Change python version if needed
    fi

    # Activate the environment
    source activate $ENV_NAME || conda activate $ENV_NAME
fi

# Install packages using conda defaults and conda-forge where necessary
echo "Installing packages using conda..."
conda install -y anaconda::numpy
conda install -y anaconda::pandas

conda install -y anaconda::scikit-learn
conda install -y conda-forge::keras

# conda install -y \
conda install -y conda-forge::defaults
conda install -y conda-forge::absl-py
conda install -y conda-forge::anyio
conda install -y conda-forge::argon2-cffi
conda install -y conda-forge::arrow
conda install -y conda-forge::asttokens
conda install -y conda-forge::async-lru
conda install -y conda-forge::attrs
conda install -y conda-forge::babel
conda install -y conda-forge::beautifulsoup4
conda install -y conda-forge::bleach
conda install -y conda-forge::cachetools
conda install -y conda-forge::certifi
conda install -y conda-forge::cffi
conda install -y conda-forge::charset-normalizer
conda install -y conda-forge::contourpy
conda install -y conda-forge::cycler
conda install -y conda-forge::debugpy
conda install -y conda-forge::decorator
conda install -y conda-forge::defusedxml
conda install -y conda-forge::exceptiongroup
conda install -y conda-forge::executing
conda install -y conda-forge::flatbuffers
conda install -y conda-forge::fonttools
conda install -y conda-forge::gast
conda install -y conda-forge::google-auth
conda install -y conda-forge::google-auth-oauthlib
conda install -y conda-forge::grpcio
conda install -y conda-forge::h5py
conda install -y conda-forge::httpcore
conda install -y conda-forge::httpx
conda install -y conda-forge::idna
conda install -y conda-forge::imageio
conda install -y conda-forge::imgaug
conda install -y conda-forge::ipykernel
conda install -y conda-forge::ipython
conda install -y conda-forge::ipywidgets
conda install -y conda-forge::jedi
conda install -y conda-forge::jinja2
conda install -y conda-forge::joblib
conda install -y conda-forge::json5
conda install -y conda-forge::jsonschema
conda install -y conda-forge::jupyter
conda install -y conda-forge::jupyter-console
conda install -y conda-forge::jupyter_client
conda install -y conda-forge::jupyter_core
conda install -y conda-forge::jupyter_server
conda install -y conda-forge::jupyterlab
conda install -y conda-forge::keras
conda install -y conda-forge::kiwisolver
conda install -y conda-forge::lazy_loader
conda install -y conda-forge::libclang
conda install -y conda-forge::markdown
conda install -y conda-forge::markdown-it-py
conda install -y conda-forge::markupsafe
conda install -y conda-forge::matplotlib-inline
conda install -y conda-forge::mdurl
conda install -y conda-forge::mistune
conda install -y conda-forge::nbclient
conda install -y conda-forge::nbconvert
conda install -y conda-forge::nbformat
conda install -y conda-forge::nest-asyncio
conda install -y conda-forge::networkx
conda install -y conda-forge::notebook
conda install -y conda-forge::opencv
conda install -y conda-forge::optree
conda install -y conda-forge::overrides
conda install -y conda-forge::packaging
conda install -y conda-forge::pillow
conda install -y conda-forge::platformdirs
conda install -y conda-forge::prometheus_client
conda install -y conda-forge::prompt_toolkit
conda install -y conda-forge::protobuf
conda install -y conda-forge::psutil
conda install -y conda-forge::pyasn1
conda install -y conda-forge::pyasn1-modules
conda install -y conda-forge::pycparser
conda install -y conda-forge::pydot
conda install -y conda-forge::pygments
conda install -y conda-forge::pyparsing
conda install -y conda-forge::python-dateutil
conda install -y conda-forge::python-json-logger
conda install -y conda-forge::pytz
conda install -y conda-forge::pyyaml
conda install -y conda-forge::pyzmq
conda install -y conda-forge::qtconsole
conda install -y conda-forge::qtpy
conda install -y conda-forge::referencing
conda install -y conda-forge::requests
conda install -y conda-forge::requests-oauthlib
conda install -y conda-forge::rich
conda install -y conda-forge::rsa
conda install -y conda-forge::scikit-image
conda install -y conda-forge::scikit-learn
conda install -y conda-forge::scipy

conda install -y anaconda::seaborn
conda install -y anaconda::send2trash
conda install -y anaconda::shapely
conda install -y anaconda::six
conda install -y anaconda::sniffio
conda install -y anaconda::soupsieve
conda install -y anaconda::tensorboard
conda install -y anaconda::tensorflow
conda install -y anaconda::tensorflow-estimator
conda install -y anaconda::termcolor
conda install -y anaconda::terminado
conda install -y anaconda::threadpoolctl
conda install -y anaconda::tifffile
conda install -y anaconda::tinycss2
conda install -y anaconda::tomli
conda install -y anaconda::tornado
conda install -y anaconda::tqdm
conda install -y anaconda::traitlets
conda install -y anaconda::typing-extensions
conda install -y anaconda::tzdata
conda install -y anaconda::urllib3
conda install -y anaconda::wcwidth
conda install -y anaconda::webencodings
conda install -y anaconda::websocket-client
conda install -y anaconda::werkzeug
conda install -y anaconda::widgetsnbextension
conda install -y anaconda::wrapt

echo "Installing additional packages from conda-forge..."
conda install -y conda-forge \
    aggdraw \
    notebook-shim \
    namex \
    jsonpointer \
    rfc3339-validator \
    rfc3986-validator

conda install -y conda-forge::jupyterlab_widgets \
    webcolors \
    stack_data \
    pure_eval \
    opt_einsum \
    ml_dtypes \
    jupyter-console \
    isoduration \
    fqdn \
    fastjsonschema \
    tensorflow-io-gcs-filesystem \
    jupyterlab_widgets

conda install -y hcc::visualkeras

conda install -y anaconda::jupyterlab_server
conda install -y conda-forge::jupyter_server_terminals \
    conda-forge::matplotlib \

conda install -y anaconda::kagglehub

echo "Installation complete. Activate the environment using:"
echo "conda activate $ENV_NAME"
