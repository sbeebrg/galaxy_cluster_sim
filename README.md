Source files for the setup of a galaxy cluster imaging model, to be used in ScopeSim_Templates.  
The cluster_setup.py file is responsible for generating physical parameters used to describe the positions and luminosity profiles of individual cluster galaxies.  
The clusters.py file uses these parameters in order to place Sérsic Profile "stamps" on a blank canvas. It returns a ScopeSim Source object containing both an image HDU and an averaged spectrum.  
The CSV file containes a lookup table of per solar mass apparent magnitudes, necessary for obtaining galaxy magnitudes through stellar population synthesis.
