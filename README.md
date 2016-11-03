# csci8360-p4-tooYoungTooSimple

#   Neuron Recognition with constrained nonnegative matrix factorization (CNMF)

This is project 4 in CSCI 8360 course at University of Georgia, Spring 2016. In this project we were challenged to identify neurons in time-series image data in the testing set. The training sets contain 19 samples and the testing sets contain 9 samples. Each sample includes a variable number of time-varying TIFF images. Each sample has unique numbers and positions of the neurons. We converted formats of the TIFF image data and created condensed movies based on them as the inputs of model. Then we applied the contrained nonnegative matrix factorization method to train the model. In the end, the model gives us averages of recall, prescision, inclusion, exclusion and combined to be 0.49, 0.40, 0.72, 0.74 and 0.39 respectively on the test set.

## Getting Started

Here are the instructions to run the python scripts to apply constrained nonnegative matrix factorization on the dataset. Be sure to change the data path when you run the scripts locally.


### Patch TIFF images and convert to a single TIF file
**run patchTiff.py**

### Configuration
**[Calcium imaging analysis toolbox](https://github.com/eds-uga/csci8360-p4-tooYoungTooSimple/tree/master/test_temp/CalBlitz)**

## Constrained nonegative matrix factorization
Here are the reference papers:

**[Simultaneous denoising, deconvolution, and demixing of calcium imaging data](http://ac.els-cdn.com/S0896627315010843/1-s2.0-S0896627315010843-main.pdf?_tid=1bc467d8-a205-11e6-8d05-00000aacb361&acdnat=1478205530_318fdf74450255ecd0e8701682903d40)**

**[A structured matrix factorization framework for large scale calcium imaging data analysis](https://arxiv.org/pdf/1409.2903v1.pdf)**


###Model Training and Testing

* Decompose the spatiotemporal activity into spatial components with local structure and temporal components that model the dynamics of the calcium by the constrained matrix factorization method and predict locations of recognized neurons by filtering out coordinates with high intensities.

###Collect recognition results

* Save results for the 9 samples in the test set by .json form. Each consists of all the coordinates surrounding predicted neurons in the sample. 

## Running
* Run neuroTest.py in the terminal.

## Authors

* **[Xiaodong Jiang](https://www.linkedin.com/in/xiaodongjiang)** - Ph.D. Student, *Department of Statistics*
* **[Yang Song](https://www.linkedin.com/in/yang-song-74298a118/en)** - M.S. Student, *Department of Statistics*
* **[Yaotong Cai](https://www.linkedin.com/in/yaotong-colin-cai-410ab026)** - Ph.D. Candidate, *Department of Statistics*

## Acknowledgments

* Thanks all team members for the laborious work and great collaboration.
* Thanks [Dr. Quinn](http://cobweb.cs.uga.edu/~squinn/).
