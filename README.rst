Neural Doodle
=============

A minimalistic implementation of Semantic Style Transfer (Champandard, 2016), based on the Neural Patches algorithm (Li 2016).

|Python Version| |License Type| |Project Stars|

----

.. image:: docs/Landscape_example.png

Image Analogy
-------------

The algorithm is built for style transfer, but it can also handle making image analogies.  Files are
included in the `#/samples/` folder. Execute with these commands:

.. code:: bash

    # Synthesize a coastline in the style of Monet. Uses "*_sem.png" files for both images.
    python3 doodle.py --style samples/Monet.jpg --output samples/Coastline.jpg

    # Generate a scene around a lake in the style of Renoir. 
    python3 doodle.py --style samples/Renoir.jpg --output samples/Landscape.jpg 

Installation & Setup
--------------------

This project requires Python 3.x. You'll also need `numpy` and `scipy` (numerical computing libraries)
installed system-wide. Afterwards, you can run the following commands from your terminal:

.. code:: bash

    # Create a local environment for Python 3.x to install dependencies here.
    python3 -m venv pyvenv --system-site-packages
    
    # If you're using bash, make this the active version of Python.
    source pyvenv/bin/activate
    
    # Setup the required dependencies simply using the PIP module.
    python3 -m pip install -r https://raw.githubusercontent.com/alexjc/neural-doodle/master/requirements.txt

.. image:: docs/Coastline_example.png

Frequest Questions
------------------

Q: How is semantic style transfer different than neural image analogies?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It's still very early too say definitively, both approaches were discovered independently at the same time by @alexjc and @awentzonline (respectively). Here are our early impressions:

1. One algorithm is style transfer that happens to do analogies, and the other is analogies that happens to do style transfer now. Adam extended his implementation to use a content loss after the semantic style transfer paper was published, so now they're very similar!

2. Both use a patch-based approach (Li, 2016) but semantic style transfer imposes a "prior" on the patch-selection process and neural analogies has an additional prior on the convolution activations.  The outputs for both algorithms are a little different, it's not yet clear where each one is best.

3. Semantic style transfer is simpler, it has fewer loss components.  This means somewhat less code to write and there are fewer parameters involved (not necessarily positive or negative).  Neural analogies is a little more complex, with as many parameters as the combination of two algorithms.

4. Neural analogies is designed to work with images, and can only support the RGB format for its masks. Semantic style transfer was designed to integrate with other neural networks (for pixel labeling and semantic segmentation), and can use any format for its maps, including RGBA or many channels per label masks.

5. Semantic style transfer is about 25% faster and uses less memory too.  For neural analogies, the extra computation is effectively the analogy prior—which could improve the quality of the results in theory. In practice, it's hard to tell—let us know what you find!

----

|Python Version| |License Type| |Project Stars|

.. |Python Version| image:: http://aigamedev.github.io/scikit-neuralnetwork/badge_python.svg
    :target: https://www.python.org/

.. |License Type| image:: https://img.shields.io/badge/license-New%20BSD-blue.svg
    :target: https://github.com/alexjc/neural-doodle/blob/master/LICENSE

.. |Project Stars| image:: https://img.shields.io/github/stars/alexjc/neural-doodle
    :target: https://github.com/alexjc/neural-doodle/stargazers
