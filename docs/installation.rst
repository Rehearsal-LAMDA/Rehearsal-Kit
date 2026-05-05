Installation
============

Base Install
------------

Install the base package with pip:

.. code-block:: bash

   python -m pip install rehearsal

The base install includes the NumPy-backed core APIs, method adapters,
experiment runner, and installed toy demo.

Optional Extras
---------------

Bermuda ``.mat`` loading needs SciPy:

.. code-block:: bash

   python -m pip install "rehearsal[bermuda]"

QWZ23 can use SciPy for the preferred MILP optimizer:

.. code-block:: bash

   python -m pip install "rehearsal[qwz23]"

Source Checkout
---------------

Install directly from the source repository:

.. code-block:: bash

   git clone https://github.com/DWB1115/Rehearsal-Kit.git
   cd Rehearsal-Kit
   python -m pip install .
