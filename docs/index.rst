Rehearsal Documentation
=======================

``rehearsal`` is a unified Python framework for rehearsal-learning research
code from LAMDA, Nanjing University. It provides shared task contracts,
structural-model interfaces, method adapters, optimizers, metrics, influence
measures, datasets, and seeded experiment runners.

Rehearsal learning was proposed by Professor Zhi-Hua Zhou to discern influence
relations for decision-making. It addresses AUF tasks: given an observed context
``X`` and a predicted undesired outcome ``Y`` outside a desired region ``S``,
the goal is to determine a decision that steers ``Y`` toward ``S``.

Guidepost
---------

* :doc:`installation` describes the base install, optional extras, and source
  checkout workflow.
* :doc:`quickstart` shows the installed smoke demo, Python API, and seeded
  experiment runner.
* :doc:`methods` lists the currently registered rehearsal-learning method
  adapters.
* :doc:`architecture` summarizes the package contracts and layer boundaries.

Installation
------------

Install the base package with pip:

.. code-block:: bash

   python -m pip install rehearsal

Quickstart
----------

Run the installed smoke demo:

.. code-block:: bash

   rehearsal-demo --seed 3 --n-samples 40 --eval-samples 6 --max-iters 5 --compact

Resources
---------

* Source code: https://github.com/DWB1115/Rehearsal-Kit
* Foundational paper: https://www.lamda.nju.edu.cn/publication/fcs22_rehearsal.pdf
* LAMDA homepage: https://www.lamda.nju.edu.cn/

.. toctree::
   :maxdepth: 2
   :caption: For Users

   installation
   quickstart
   methods

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   architecture
   experiments

.. toctree::
   :maxdepth: 2
   :caption: For Developers

   api
   contributing

.. toctree::
   :maxdepth: 2
   :caption: About

   reference
   about
