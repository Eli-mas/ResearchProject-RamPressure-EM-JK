# ResearchProject-RamPressure-EM-JK
Repository to track the development of an astrophysical research project conducted with [Jeffrey Kenney](http://www.astro.yale.edu/kenney/pages/index.html) of Yale's astronomy department.

## A note on the status: Sept 2024
Unfortunately this project ceased activity in the Summer of 2021, and there are no plans to continue with it.

## Project aims and focus
The purpose of the project is to develop a simple yet objective & reliable method for detecting the presence of ram pressure acting on a galaxy, and then estimating the direction of this pressure over the galaxy, given only a moment-0 map of the galaxy's gas content (HI [neutral hydrogen] for observational data, mixed for simulated data) at a particular time. We then seek to understand how our method performs on various galaxies for which we have data.

Our method offers novel ways of quantifying asymmetries in the outer regions of a galaxy, and from this we estimate whether a galaxy's asymmetry offers sufficient evidence for ram pressure, and, if so, the direction of the ram pressure's origin, which implicitly yields an estimate of the galaxy's direction of motion.

## (From March 2021) Current state of the project:
Articles I & II are written; we are tidying them up before submitting to the journal. Articles III & IV are mostly written, current focus is on refining figures. Targeting [ApJ](https://iopscience.iop.org/journal/0004-637X) for submission.

A fifth paper is partially underway that collects what we have learned in light of galaxy evolution from the application of our methods to observed galaxies.

These five papers all stem from the methods that will be presented in the first paper. We have also developed more complex methods of measuring asymmetry that can be detailed in other articles.

## This repository
I have posted the code here that pertains to the earlier papers. There is some dependency on [this module](https://github.com/Eli-mas/common), but not to an extent that prevents understanding the code in this repository.

The overview of what is here:
* `asy_io`: miscellaneous functions relating to input and output.
* `cls`: classes that make for convenient scripting and utility across the project.
    - `analyze`: functions pertinent to the GalaxySeries class grouped modularly by general purpose.
    - `classes`: classes that provide an interface to data loading, storage, and computation
        + `GalaxyCollection`: classes that aggregate across Galaxy objects. For simulation data, the contained GalaxySeries class is central. Note: some features of this module have been deprecated by `GalaxySets`
        + `h5`: fully encapsulating interface to h5py API
        + `Galaxy`: The core of the project; provides a simple interface to access, store, and compute on data pertinent to an individual galaxy. The contained `Galaxy.py` file defines the Galaxy class, and other files help to organize the logic of this class.
        + `GalaxySets`: useful mostly for aggregating/plotting data across various samples of galaxies. Do you like metaclasses and polymorphism?
    - `plotters`: modules that contain functions meant to associate with the `Galaxy` or `GalaxySeries` classes. Modules are named according to the associations and purposes their functions serve.
* `comp`: computational functions for different aspects of the project
    - `array_functions.py`: lower-level array manipulation routines; much of the original functionality has been moved to [another repository](https://github.com/Eli-mas/common) of mine.
    - `asymmetry_functions.py`: the project revolves around calculating m=1/m=2 asymmetries; this script encodes the procedures for doing this for any provided galaxy.
    - `computation_functions.py`: miscellaneous computation functions.
    - `contiguous.py`: Moment-0 maps, the raw data source, provide flux data across an array of pixels. But not all of these pixels involve the galaxy of interest--and some maps contain more than one galaxy. Calculating a galaxy's asymmetry requires assignment of pixels to the galaxy, without capturing other pixels. Our decision is that only pixels contiguous with the galaxy's center coordinate are to be considered for analysis, and this module encapsulates the logic of isolating these pixels--hence its name.
    - `polar_functions.py`: routines to handle analysis of periodic data, i.e. data bounded on a finite interval in some modular space--e.g., calculating aggregate statistics on such data.
    - `smoothing.py`: as with standard photography, astrophysical images come in different resolutions. We want those resolutions to be standard to the extent possible before analysis. For data that is of too high resolution, this is accomplished by smoothing over-resolved image data--hence the name of this module.
* `plot`: incorporates functions to visualize data and make plots for publications. Many of these are used in the `papers` module.
* `prop`: basic quantities and variable definitions. The scripts in this module are mostly for defining the galaxies that are used elsewhere in the project.

## Usage
This repository is posted to GitHub for reference; it is not meant to be downloaded to another machine as a plug-and-play package as of yet, though it should not take too much more work to get it to that state if desired. The purpose of having this code up is to allow for inspection, and I have tried to put up all code that is critical to understanding what is going on in the first set of papers.
