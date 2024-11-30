############
Installation
############

There are many ways you can install our package, using pip to download our latest stable version,
Docker, Singularity/Apptainer or manual installation directly from the source code. We highly suggest to install
our latest PyPI wheel. Note that the Singularity/Apptainer versions are outdated.


****************
Install with pip
****************

To install **NiChart DLMUSE** with pip, just do: ::

    $ pip install NiChart_DLMUSE

We always have our latest stable version on PyPI, so we highly suggest you to install it this way, as this package is under
heavy development and building from source can lead to crashes and bugs.


.. _`Docker Container`:

****************
Docker Container
****************

The package comes already pre-built as a docker container that you can download at our `docker hub <https://hub.docker.com/r/cbica/nichart_dlmuse/tags>`_.
You can build the package by running the following command: ::

    $ docker build -t cbica/nichart_dlmuse .

.. _`Singularity/Apptainer build`

Singularity and Apptainer images can be built for NiChart_DLMUSE, allowing for frozen versions of the pipeline and easier
installation for end-users. Note that the Singularity project recently underwent a rename to "Apptainer", with a commercial
fork still existing under the name "Singularity" (confusing!). Please note that while for now these two versions are largely identical,
future versions may diverge. It is recommended to use the AppTainer distribution. For now, these instructions apply to either.
After installing the container engine, run: ::

    $ singularity build nichart_dlmuse.sif singularity.def

This will take some time, but will build a containerized version of your current repo. Be aware that this includes any local changes!
The nichart_dlmuse.sif file can be distributed via direct download, or pushed to a container registry that accepts SIF images.

.. _`Manual installation`

You can manually build the package from source by running: ::

    $ git clone https://github.com/CBICA/NiChart_DLMUSE

    $ cd NiChart_DLMUSE && python3 -m pip install -e .

We **do not** recomment installing the package directly from source as the repository above is under heavy development and can cause
crashes and bugs.
