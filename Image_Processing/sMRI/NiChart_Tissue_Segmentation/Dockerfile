FROM aidinisg/dlicv:0.0.0

RUN apt-get update && apt-get install -y wget git && \
    wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
    python fslinstaller.py -d /usr/local/fsl && \
    rm fslinstaller.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV FSLDIR=/usr/local/fsl
ENV PATH=$FSLDIR/bin:$PATH
ENV FSLOUTPUTTYPE=NIFTI_GZ

RUN cd / && \
    git clone https://github.com/CBICA/NiChart_Tissue_Segmentation && \
    cd NiChart_Tissue_Segmentation/ && \
    pip install . && \
    cp -r /DLICV/model /NiChart_Tissue_Segmentation/ 

# Set the default command or entrypoint (optional, depending on your package needs)
ENTRYPOINT ["NiChart_Tissue_Segmentation", "--model", "/NiChart_Tissue_Segmentation/model/"]
WORKDIR /workspace/
CMD ["--help"]