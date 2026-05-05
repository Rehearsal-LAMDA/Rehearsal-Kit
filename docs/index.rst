Rehearsal
=========

``rehearsal`` is a unified Python framework for rehearsal-learning research
code from LAMDA, Nanjing University.

Rehearsal learning was proposed by Professor Zhi-Hua Zhou to discern influence
relations for decision-making. It addresses AUF tasks: given an observed context
``X`` and a predicted undesired outcome ``Y`` outside a desired region ``S``,
the goal is to determine a decision that steers ``Y`` toward ``S``.

The package provides shared task contracts, structural-model interfaces, method
adapters, optimizers, metrics, influence measures, datasets, and seeded
experiment runners for comparing rehearsal-learning methods under one command
line shape.

.. code-block:: bash

   python -m pip install rehearsal

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   methods
   reference
   architecture

Project Links
-------------

* Source code: https://github.com/DWB1115/Rehearsal-Kit
* Foundational paper: https://www.lamda.nju.edu.cn/publication/fcs22_rehearsal.pdf
