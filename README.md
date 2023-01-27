<!-- ![pic1](eccojams_text_logo.png) -->
<p align="center">
  <img width="80%" src="eccojams_text_logo.png">
</p>

# ECCOJAMS

Jams you will return to again and again...

A set of tools for analyzing neural data after use of musclebeachtools.  Also included
are utility functions for various steps in analysis
of chronic neural and behavioral data.

Find a tutorial in:
```
tutorial/eccojams_tutorial.ipynb
```

## Installation
```
git clone https://github.com/hengenlab/eccojams.git
cd <location_of_eccojams>/eccojams/
pip install .
```

## Unzip example data for tutorial
```
cd <location_of_eccojams>/eccojams/tutorial/example_data/
unzip example_data.zip
```

## Test import
```
Open powershell/terminal
    ipython
    import eccojams as eco
```

## What's new?
```
1-27-23:
Now can select any event times by sleep state (not just ripples).
Example usage:

    import numpy as np
    import eccojams as eco

    sleepdir = '/media/HlabShare/Sleep_Scoring/CAF00077/02022021/'

    events = np.random.uniform(0,24*3600,1000)
    sleepdf = eco.return_sleep_scores(sleepdir + 'CAF77_consensus.npy')

    eco.eventtimes_by_state(events, sleepdf, 'nrem')
```
