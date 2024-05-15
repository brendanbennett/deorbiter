.. Mir Orbiter documentation master file, created by
   sphinx-quickstart on Tue May 14 01:33:42 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Mir Orbiter's documentation!
=======================================

Mir Orbiter is a Python package that provides simulation, prediction and analysis tools for satellite orbits. 
It is designed to be easy to use and flexible, and is suitable for a wide range of applications.

This project is the collaborative effort of six team members as part of our group coursework. 
The primary objective of this assignment is to develop a complex piece of predictive modeling research software 
that implements advanced methods and adheres to software design principles learned throughout our course. Our 
work addresses a cross-discipline research computing challenge: predicting the de-orbiting trajectory of a satellite 
and its eventual impact point on Earth.

It explores the use of an Extended Kalman Filter (EKF) combined with a simulation engine to accurately 
predict the trajectory and impact location of deorbiting satellites. This approach addresses the risks posed by 
uncontrolled satellite reentry by leveraging simulated radar measurements and handling nonlinearities in the 
system. By improving prediction accuracy and minimizing uncertainty, the project aims to enhance preventative 
measures against potential satellite debris impacts.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   simulator
   predictor
   observer
   plotting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
