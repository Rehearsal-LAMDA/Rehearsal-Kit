Rehearsal Documentation
=======================

``rehearsal`` is a unified Python framework for rehearsal-learning research,
developed by the LAMDA Group at Nanjing University. It provides shared task contracts,
structural-model interfaces, method adapters, optimizers, metrics, influence
measures, datasets, and seeded experiment runners.

Rehearsal learning was proposed by Professor Zhi-Hua Zhou from Nanjing University,
aiming to discern influence relations, a type of relation tailored for decision-making.
Rehearsal learning is introduced to address AUF tasks: given an observed context ``X``
and a predicted undesired outcome ``Y`` falling outside a pre-specified desired region ``S``,
the goal is to determine the decision to steer ``Y`` toward ``S``. See the
`reference PDF <https://www.lamda.nju.edu.cn/publication/fcs22_rehearsal.pdf>`_.

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

* Source code: Gitee 主页 https://gitee.com/Rehearsal-LAMDA/rehearsal-kit | GitHub 主页 https://github.com/Rehearsal-LAMDA/Rehearsal-Kit
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
