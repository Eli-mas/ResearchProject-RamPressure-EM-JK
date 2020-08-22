# ResearchProject-RamPressure-EM-JK
Repository to track the development of an astrophysical research project conducted with [Jeffrey Kenney](http://www.astro.yale.edu/kenney/pages/index.html) of Yale's astronomy department. I have personally been on this project since June 2016.

__*Most recent update: (8-15-20)*__

## Project aims and focus
The purpose of the project is to develop a simple yet objective & reliable method for detecting the presence of ram pressure acting on a galaxy, and then estimating the direction of this pressure over the galaxy, given only a moment-0 map of the galaxy's gas content (HI [neutral hydrogen] for observational data, mixed for simulated data) at a particular time. We then seek to understand how our method performs on various galaxies for which we have data.

Our method offers novel ways of quantifying asymmetries in the outer regions of a galaxy, and from this we estimate whether a galaxy's asymmetry offers sufficient evidence for ram pressure, and, if so, the direction of the ram pressure's origin, which implicitly yields an estimate of the galaxy's direction of motion.

## Current state of the project:
We currently have three articles mostly written that will present the new methodology developed, and will offer an evaluation of the results generated on two sets of data: a suite of time-evolving galaxy simulations, and a batch of real (observed) galaxies from nearby sources where HI moment-0 maps are available. We are currently tidying up the papers to send to co-authors for review prior to subimtting to the Astrophysical Journal (ApJ). The intention is to submit for review this Fall.

A fourth paper is partially drafted that details the overview of our motion estimates for the nearby Virgo cluster. A fifth paper is planned (analysis has been at least partially conducted) that collects what we have learned in light of galaxy evolution from the application of our methods to observed galaxies.

These five papers all stem from the methods that will be presented in the first paper. We have also developed more complex methods of measuring asymmetry that may be detailed in other articles.

## This repository
I have posted the code here that pertains to the earlier papers, particularly the first three, with some miscellaneous scripting functions included. Code for potential future papers is not posted yet. Parts of this project depend on other modules that I have developed on my local machine; I will post parts of these in time.

Components that allow the scripts to work but do not contribute to an understanding of the paper's methods, such as the file asy_io/asy_paths.py (which only defines paths to particular directories and files), are not included. A number of plotting routines are also not included yet (they will be closer to publication).

The overview of what is here:
* *asy_io*: miscellaneous functions relating to input and output.
* *cls*: classes that make for convenient scripting and utility across the project.
* *comp*: computational functions for different aspects of the project
* *plot*: plotting functions, a few files are not yet included
* *prop*: basic quantities and variable definitions. The scripts in this module are mostly for defining the galaxies that are used elsewhere in the project; I have left most out for the moment to reduce clutter.

## Usage
This repository is posted to GitHub for reference; it is in no state to be downloaded to another machine as a plug-and-play package. The purpose of having this code up is to allow for open inspection, and I have tried to put up all code that is critical to understanding what is going on in the first few papers.

The state of comments in the code is variable--I am working on this. There are also some points of general maintenance to be handled, such as removing obsolete functions, but for the most part what is in these files is relevant to the current work.

When we start publishing I can shift gears to develop a proper, all-batteries-included software package out of this if there is interest in the research community. The more likely scenario is that I will develop something out of the generalized routines I have developed over the course of this research, e.g. in the [common module](https://github.com/Eli-mas/common).
