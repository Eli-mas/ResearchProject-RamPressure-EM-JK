# ResearchProject-RamPressure-EM-JK
Repository to track the development of an astrophysical research project conducted with [Jeffrey Kenney](http://www.astro.yale.edu/kenney/pages/index.html) of Yale's astronomy department.

## My role in the project
* **Software development**: Since starting, I have been the sole developer of this code base. I expanded it from a simple set of Python scripts into a series of interwoven modules aimed at being more functional, extensible, efficient software; this included implementing a wrapper around the HDF5 Python bindings (h5py), and employing parallelization and the scientific Python ecosystem (numpy, scipy, pandas, numba) to execute desired algorithms.
* **Efficiency and scope of analysis**: The original implementation required ~20-30 minutes to analyze a set of ~50 galaxies, each of which was relatively small in size. Given my improvements, our current dataset, comprising ~1350 galaxies (many of which are considerably larger than those in the original sample), runs through the primary analysis in 1-2 minutes, while executing additional algorithms beyond the original implementation.
* **Researcher**: I have conceived of and implemented different analytical methods and analyzed results to answer research questions, the most fundamental of which is how to most effectively trace the presence of ram pressure by asymmetries introduced into galaxy morphology. To this end, I developed several more advanced methods for measuring asymmetry that utilize more information about the gas distribution than the baseline method I was presented with at the start of the project; my experimentation extended the project’s publishable results well beyond the original scope.
* **Authorship/collaboration**: I am first author on the most analytically intensive article of the initial batch (paper II), and co-author on others. Dr. Kenney and I have collaborated closely to interpret results, conceive of data visualizations that would help our analysis, and generate new analytical ideas.

## Project aims and focus
The purpose of the project is to develop a simple yet objective & reliable method for detecting the presence of ram pressure acting on a galaxy, and then estimating the direction of this pressure over the galaxy, given only a moment-0 map of the galaxy's gas content (HI [neutral hydrogen] for observational data, mixed for simulated data) at a particular time. We then seek to understand how our method performs on various galaxies for which we have data.

Our method offers novel ways of quantifying asymmetries in the outer regions of a galaxy, and from this we estimate whether a galaxy's asymmetry offers sufficient evidence for ram pressure, and, if so, the direction of the ram pressure's origin, which implicitly yields an estimate of the galaxy's direction of motion.

## Current state of the project:
Articles I & II are written and have been sent to co-authors for review. Articles III & IV are mostly written, current focus is on refining figures. Expecting to submit these all to [ApJ](https://iopscience.iop.org/journal/0004-637X) for review by January 2021.

A fifth paper is partially underway that collects what we have learned in light of galaxy evolution from the application of our methods to observed galaxies.

These five papers all stem from the methods that will be presented in the first paper. We have also developed more complex methods of measuring asymmetry that may be detailed in other articles.

## This repository
I have posted the code here that pertains to the earlier papers, particularly the first three, with some miscellaneous scripting functions included. Code for potential future papers is not posted yet. Parts of this project depend on other modules that I have developed on my local machine; I will post parts of these in time.

Components that allow the scripts to work but do not contribute to an understanding of the paper's methods, such as files defining paths to other files on the machine, are not included. A number of plotting routines are also not included yet.

The overview of what is here:
* `asy_io`: miscellaneous functions relating to input and output.
* `cls`: classes that make for convenient scripting and utility across the project.
    - `analyze`: functions pertinent to the GalaxySeries class grouped modularly by general purpose
    - `classes`: classes that provide an interface to data loading, storage, and computation
        + `GalaxyCollection`: classes that aggregate across Galaxy objects. For simulation data, the contained GalaxySeries class is central. Note: some features of this module have been deprecated by `GalaxySets`
        + `h5`: fully encapsulating interface to h5py API
        + `Galaxy`: The core of the project; provides a simple interface to access, store, and compute on data pertinent to an individual galaxy. The contained `Galaxy.py` file defines the Galaxy class, and other files help to organize the logic of this class.
        + `GalaxySets`: useful mostly for aggregating/plotting data across various samples of galaxies. Do you like metaclasses and polymorphism?
    - `plotters`: modules that contain functions meant to associate with the `Galaxy` or `GalaxySeries` classes. Modules are named according to the associations and purposes their functions serve.
* `comp`: computational functions for different aspects of the project
    - `array_functions.py`: lower-level array manipulation routines; somewhat miscllaneous, should probably be refactored.
    - `asymmetry_functions.py`: the project revolves around calculating m=1/m=2 asymmetries; this script encodes the procedures for doing this for any provided galaxy.
    - `computation_functions.py`: miscellaneous computation functions; should also be refactored, probably merged with `array_functions.py`.
    - `contiguous.py`: Moment-0 maps, the raw data source, provide flux data across an array of pixels. But not all of these pixels involve the galaxy of interest--and some maps contain more than one galaxy. Calculating a galaxy's asymmetry requires assignment of pixels to the galaxy, without capturing other pixels. Our decision is that only pixels contiguous with the galaxy's center coordinate are to be considered for analysis, and this module encapsulates the logic of isolating these pixels--hence its name.
    - `polar_functions.py`: routines to handle analysis of periodic data, i.e. data bounded on a finite interval in some modular space--e.g., calculating aggregate statistics on such data.
    - `smoothing.py`: as with standard photography, astrophysical images come in different resolutions. We want those resolutions to be standard to the extent possible before analysis. For data that is of too high resolution, this is accomplished by smoothing over-resolved image data--hence the name of this module.
* `plot`: incorporates functions to visualize data and make plots for publications. Many of these are used in the `papers` module.
* `prop`: basic quantities and variable definitions. The scripts in this module are mostly for defining the galaxies that are used elsewhere in the project.

## Usage
This repository is posted to GitHub for reference; it is not meant to be downloaded to another machine as a plug-and-play package as of yet. The purpose of having this code up is to allow for open inspection, and I have tried to put up all code that is critical to understanding what is going on in the first set of papers.