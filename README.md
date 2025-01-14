# Intra-individual circadian rhythm average and variability

This is the code accompanying _"More variable circadian rhythms in epilepsy captured by long-term heart rate recordings from wearable sensors"_ (Smith et. al, 2024), which is currently in preprint @ https://doi.org/10.48550/arXiv.2411.04634.

This code specifically has been run with Python 3.11, however should be compatible with any reasonably modern Python 3 version. 
Please see requirements.txt for a list of the required packages and their dependencies. 

We recommend creating a virtual environment and then installing the requirements before running this code:
- `python -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

Please do not hesitate to get in touch at b.smith16@newcastle.ac.uk with any technical issues, questions or concerns. 

There are two main scripts:
- _produce_publication_figures.py_
  1. loads-in _data/segments_df.csv_, **which is the actual data used for the main figures in our publication**.
     - we are unable to share the raw data used to produce this, but see the script below
  2. produces figures 3-5 (results 1-3) contained in the publication.
  
- _intra_individual_variability_pipeline.py_ (**WIP**)
  - this is a simplified reproduction of the pipeline we used to calculate the table above, over simulated data.
  - this uses some generated synthetic data (see _generate_synthetic_data.py_), but **you can also pass your own time-series data to this method**. 
  
  


## _produce_publication_figures.py_

> ### segments_df
> Each row of this data frame corresponds to a seven-day segment for a given participant.
> - _Participant_: the participant the segment belongs to _(str)_
> - _Cohort_: the cohort the participant belongs to ('control' or 'PWE') _(str)_
> - _Run ID_: the index of the 'run' that segment belongs to within the participant's recording (chronologically ordered) _(int)_
> - _Segment ID_: the index of the segment within the run (chronologically ordered) _(int)_
> - _Period/Acrophase/Amplitude Mean_: the mean of that circadian property within this segment _(float)_
> - _Period/Acrophase/Amplitude Std_: the standard deviation of that circadian property within this segment _(float)_
> - _N Seizures_: the number of seizures within this segment _(int)_
> - _In Seizure Diary_: whether or not this segment overlaps with the participant's seizure diary period _(bool)_
> - _Duration (h)_: the duration, in hours, of this segment (int)
> - _Participant N Segments_: the number of segments for this participant overall (int)