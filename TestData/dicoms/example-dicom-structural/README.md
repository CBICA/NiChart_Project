# Example DICOM dataset of a structural T1-weighted scan

These data can be useful for tests and demos. Enjoy!

This dataset is a minimal excerpt of one aspect of the studyforrest
dataset published in:

    Hanke, M., Baumgartner, F. J., Ibe, P., Kaule, F. R., Pollmann, S.,
    Speck, O., Zinke, W. & Stadler, J. (2014). A high-resolution
    7-Tesla fMRI dataset from complex natural stimulation with an audio
     movie. Scientific Data, 1:140003.

  http://www.nature.com/articles/sdata20143


## Anonymization

This DICOM dataset has been created via nifti2dicom from a de-faced NIfTI
file. DICOM header fields have been set from the original DICOM files
the NIfTI image was created from. Subsequently, DICOM header were
anonymized, and certain field values have been reset using the following
command

    gdcmanon --dumb --remove 400,500 --remove 12,62 --remove 12,63 \
      --replace 0010,0010,Jane_Doe \
      --replace 0010,0020,02 \
      --replace 0008,1030,Hanke_Stadler^0024_transrep \
      --replace 0008,103E,anat-T1w \
      --replace 0018,1030,anat-T1w \
      --replace 0010,0040,F \
      --replace 0010,1010,42 \
      --replace 0010,1030,75 \
      --replace 0010,0030,19660101 \
      --replace 0020,0010,433724515 \
      --replace 0008,0022,20130717 \
      --replace 0008,0023,20130717 \
      --replace 0008,0033,142035.93 \
      -i ... -o ...

## Acknowledgements

This work was supported by the German Federal Ministry of Education and
Research (BMBF 01GQ11112), the German federal state of Saxony-Anhalt and the
European Regional Development Fund (ERDF), Project: Center for Behavioral Brain
Sciences, Imaging Platform.

