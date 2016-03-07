Neural Doodle
=============



|Python Version| |License Type| |Project Stars|

----

.. image:: docs/Landscape.png

Image Analogy
-------------

.. code:: bash

    # Synthesize a coastline in the style of Monet. Uses "*_sem.png" files for both images.
    python3 doodle.py --style samples/Monet.jpg --output samples/Coastline.jpg

    # Generate a scene around a lake in the style of Renoir. 
    python3 doodle.py --style samples/Renoir.jpg --output samples/Landscape.jpg 

Installation & Setup
--------------------

.. code:: bash

    python3 -m venv pyvenv --system-site-packages
    source pyvenv/bin/activate
    python3 -m pip install -r https://raw.githubusercontent.com/alexjc/neural-doodle/master/requirements.txt

.. image:: docs/Coastline.png

----

|Python Version| |License Type| |Project Stars|

.. |Python Version| image:: http://aigamedev.github.io/scikit-neuralnetwork/badge_python.svg
    :target: https://www.python.org/

.. |License Type| image:: https://img.shields.io/badge/license-New%20BSD-blue.svg
    :target: https://github.com/alexjc/neural-doodle/blob/master/LICENSE

.. |Project Stars| image:: https://img.shields.io/github/stars/alexjc/neural-doodle
    :target: https://github.com/alexjc/neural-doodle/stargazers
