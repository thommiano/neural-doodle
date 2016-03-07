Neural Doodle
=============



|Python Version| |License Type| |Project Stars|

----

.. image:: docs/Landscape.png

Image Analogy
-------------

.. code: bash

    # Generate a scene around a lake in the style of Renoir. 
    python3.4 doodle.py --output samples/Landscape.jpg --content-weight=0.0 --style samples/Renoir.jpg
    
    # Synthesize a coastline in the style of Monet. 
    python3.4 doodle.py --output samples/Coastline.jpg --content-weight=0.0 --style samples/Monet.jpg

Installation & Setup
--------------------

.. code: bash

    python3 -m venv pyvenv --system-site-packages
    source pyvenv/bin/activate
    python3 -m pip install -r https://raw.githubusercontent.com/alexjc/neural-doodle/master/requirements.txt

.. image:: docs/Coastline.png

----

|Python Version| |License Type| |Project Stars|

.. |Python Version| image:: http://alexjc.github.io/neural-doodle/badge_python.svg
    :target: https://www.python.org/

.. |License Type| image:: https://img.shields.io/badge/license-New%20BSD-blue.svg
    :target: https://github.com/alexjc/neural-doodle/blob/master/LICENSE

.. |Project Stars| image:: https://img.shields.io/github/stars/aigamedev/scikit-neuralnetwork.svg
    :target: https://github.com/alexjc/neural-doodle/stargazers    
