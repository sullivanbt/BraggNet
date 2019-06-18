# BraggNet

BraggNet is a proof of concept in using neural networks to integrate Bragg peaks for single crystal measurements.  In particular, it is a way to integrate Bragg peaks from TOF crystallography data collected on the [MaNDi beamline](https://neutrons.ornl.gov/mandi) at the Spallation Neutron Source (Oak Ridge National Laboratory, Oak Ridge, TN, USA).  The work is detailed in two publications, which are currently in press:

- Sullivan, B., Archibald, R., Azadmanesh, J., Vandavasi, V., Langan, P.S., Coates, L., Lynch, V., Langan, P. (2019). BraggNet: Integrating Bragg Peaks Using Neural Networks. J. Appl. Cryst. (accepted).

- Sullivan B., Archibald, R., Vandavassi, V., Langan, P.S., Coates, L., Lynch, V. (2019). Volumetric Segmentation via Neural Networks Improves Neutron Crystallography Data Analysis. Proceedings of High-Performance Machine Learning 2019 (IEEE CCGrid2019 Workshop, in press).

Unfortunately, I left ORNL before having time to flesh this out to an easily usable package.  In its current state this project is a set of scripts, some of which are sample specific.  Despite the mess, I hope this repository will provide a strong starting point for researchers interesting in applying neural networks to their own Bragg peak integration.  A description of the files is below:

- `mltools.py` - helper functions for generating training data, creating the network, and training.  This file probably contains the most interesting snippets for researchers looking to apply machine learning to their own integration routines.

- `unet_keras.py`: Script to actually train a model.

- `BVGFitTools.py` and `ICCFitTools.py`: libraries for profile fitting when generating training data. These are the same as are included with [Mantid](https://www.mantidproject.org/Main_Page), which will also need to be installed to run these scripts.

- `generate_training_data_v2*py`: Scripts which runs through the runs for a given crystal and generates training peaks from the strong peaks.  `generate_training_data_v2.py` is from the J. Appl. Cryst. paper while `generate_training_data_v2_betalac*py` are from the HPML paper.

- `generate_training_data_parallel.py`: Script to deploy multiple runs of `generate_training_data*py` at the same time.

- `integrate_peak_set_keras*py`: Scripts that load a trained model and do integration.  Similar to `generate_training_data*py`, the scripts with betalac in the title correspond to the HPML paper.

- `make_figures.py`: Helper script to make figures.

- `pySlice.py`: Helper class to visualize 3D data as 2D planes which can be scrolled through using the mouse wheel.


