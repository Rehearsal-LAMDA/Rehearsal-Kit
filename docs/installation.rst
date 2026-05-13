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

   git clone https://gitee.com/Rehearsal-LAMDA/rehearsal-kit.git
   cd rehearsal-kit
   python -m pip install .

Gitee 主页: https://gitee.com/Rehearsal-LAMDA/rehearsal-kit

GitHub 主页: https://github.com/Rehearsal-LAMDA/Rehearsal-Kit
