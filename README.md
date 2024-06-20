# GANrec: A GAN-based Data Reconstruction Framework

# Overview

GANrec is an data reconstruction framework that harnesses the power of Generative Adversarial Networks (GANs). While traditional reconstruction methods primarily rely on intricate algorithms to piece together fragmented data, GANrec employs the generative capabilities of GANs to reimagine and revitalize data reconstruction.

Originally designed for the fields of tomography and phase retrieval, GANrec shines in its adaptability. With a provision to input the forward model, the framework can be flexibly adapted for complex data reconstruction processes across diverse applications.

# Features

1. GAN-powered Reconstruction: At its core, GANrec employs GANs to assist in the reconstruction process, enabling more accurate and efficient results than conventional methods.
2. Specialized for Tomography & Phase Retrieval: GANrec has been optimized for tomography and phase retrieval applications, ensuring precision and reliability in these domains.
3. Modular Design: The framework's architecture allows users to provide their forward model, making it adaptable for various complex data reconstruction challenges.
4. Efficient and Scalable: Built to handle large datasets, GANrec ensures that speed and efficiency are maintained without compromising the accuracy of reconstruction.

# Installation

Installation

1. For the general users:
   - Create a Conda Environment:
   Create a new conda environment named ganrec.

   `conda create --name ganrec python=3.11`

   - Activate the Conda Environment:
   Activate the newly created ganrec environment.
   
   `conda activate ganrec`

   - Install from Pypi:
   
   `pip install ganrec`


2. If you want to work for some developments based on GANrec, please follow the steps below to install and set up GANrec:
   - Create a Conda Environment:
   Create a new conda environment named ganrec.
   
   `conda create --name ganrec python=3.11`

   - Activate the Conda Environment:
   Activate the newly created ganrec environment.
   
   `conda activate ganrec`
   - Clone the GANrec Repository:
   Clone the GANrec repository from GitHub to your local machine.
   
   `git clone https://github.com/XYangXRay/ganrec.git`
   
   - Install the Required Packages:
   Navigate to the main directory of the cloned repository and install the necessary packages.
   `cd ganrec`
   `python3 -m pip install -e .`

# Examples

GANrec currently has the applications for tomography reconstructon and in-line phase contrast (holography) phase retrieval:

1. X-ray tomography reconstruction:
   - [Tomography Example](https://github.com/XYangXRay/ganrec/blob/main/examples/tomography_tf.ipynb)
2. Holography phase retreival:
   - [Phase retrieval Example](https://github.com/XYangXRay/ganrec/blob/main/examples/holography_tf.ipynb)

# References

If you find GANrec is useful in your work, please consider citing:

J. Synchrotron Rad. (2020). 27, 486-493.
Available at: https://doi.org/10.1107/S1600577520000831
