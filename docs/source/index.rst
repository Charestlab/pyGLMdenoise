.. glmdenoise documentation master file, created by
   sphinx-quickstart on Tue Mar  5 11:38:14 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

************
pyGLMdenoise
************

Overview
========

The easiest way to install pyGLMdenoise is with pip:

.. code-block:: sh

    pip install glmdenoise

If your data is in BIDS format and you've preprocessed with fmriprep, 
you can run glmdenoise on your data as a terminal command:

.. code-block:: sh

    glmdenoise /my/data/dir

Or try it on a public dataset by using its datalad URL:

.. code-block:: sh

    glmdenoise ///workshops/nih-2017/ds000114

You can also use GLMdenoise directly from python:

.. code-block:: python

    from glmdenoise import GLMdenoise
    gd = GLMdenoise()
    gd.fit(design, data, tr=2.0)
    gd.plot_figures()

Table of Contents
=================

.. toctree::
   :maxdepth: 1

   console
   api

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


