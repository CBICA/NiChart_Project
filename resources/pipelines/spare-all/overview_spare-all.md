### SPARE Scores (excluding CVM)

Comprehensive pipeline to generate all SPARE scores that could be calculated from just T1w head MRI images at once.

##### List of SPARE scores:
- BA: Brain Age
- AD: Alzhemier's disease
- PSY: Psychosis
- MDD: Major Depression Disorder

#### Input

- T1-weighted head MRI scans (Nifti)
- Entry or a CSV file containing:
	- Demographics (Age, Sex)
	- Cognitive Status (IsCN)


#### Output

- All available SPARE scores excluding the CVM (Cardiovascular and metabolic disease) scores (CSV)

