Neural Doodle
=============

Use a deep neural network to borrow the skills of real artists and turn your two-bit doodles into masterpieces! This project is an implementation of `Semantic Style Transfer <http://arxiv.org/abs/1603.01768>`_ (Champandard, 2016), based on the `Neural Patches <http://arxiv.org/abs/1601.04589>`_ algorithm (Li 2016).

The ``doodle.py`` script generates an image by using three or four images as inputs: the original style and its annotation, and a target content image (optional) with its annotation (a.k.a. your doodle). The algorithm then extracts annotated patches from the style image, and incrementally transfers them over to the target image based on how closely they match.

**NOTE**: This project is possible thanks to the `nucl.ai Conference <http://nucl.ai/>`_ on **July 18-20**. Join us in **Vienna**!

|Python Version| |License Type| |Project Stars|

----

.. image:: docs/Landscape_example.png

Image Analogy
-------------

The algorithm is built for style transfer, but can also generate image analogies that we call a ``#NeuralDoodle``.  Example files are included in the ``#/samples/`` folder. Execute with these commands:

.. code:: bash

    # Synthesize a coastline as if painted by Monet. This uses "*_sem.png" masks for both images.
    python3 doodle.py --device=cpu --style samples/Monet.jpg --output samples/Coastline.jpg

    # Generate a scene around a lake in the style of a Renoir painting.  
    python3 doodle.py --device=gpu0 --style samples/Renoir.jpg --output samples/Landscape.jpg 

Note the ``--device`` argument that lets you specify which GPU or CPU to use. The default is to use ``cpu``, if you have NVIDIA card setup with CUDA/CUDNN already try ``gpu0``.

Installation & Setup
--------------------

This project requires Python 3.x. You'll also need ``numpy`` and ``scipy`` (numerical computing libraries)
installed system-wide. Afterwards, you can run the following commands from your terminal:

.. code:: bash

    # Create a local environment for Python 3.x to install dependencies here.
    python3 -m venv pyvenv --system-site-packages
    
    # If you're using bash, make this the active version of Python.
    source pyvenv/bin/activate
    
    # Setup the required dependencies simply using the PIP module.
    python3 -m pip install -r https://raw.githubusercontent.com/alexjc/neural-doodle/master/requirements.txt

.. image:: docs/Coastline_example.png

Frequent Questions
------------------

Q: How is semantic style transfer different to neural analogies?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It's still too early to say definitively, both approaches were discovered independently in 2016 by `@alexjc <https://twitter.com/alexjc>`_ and `@awentzonline <https://twitter.com/awentzonline>`_ (respectively). Here are some early impressions:

1. One algorithm is style transfer that happens to do analogies, and the other is analogies that happens to do style transfer now. Adam extended his implementation to use a content loss after the `Semantic Style Transfer <http://arxiv.org/abs/1603.01768>`_ paper was published, so now they're even more similar under the hood!

2. Both use a `patch-based approach <http://arxiv.org/abs/1601.04589>`_ (Li, 2016) but semantic style transfer imposes a "prior" via the patch-selection process and neural analogies has an additional prior on the convolution activations.  The outputs for both algorithms are a little different, it's not yet clear where each one is best.

3. Semantic style transfer is simpler, it has fewer loss components.  This means somewhat less code to write and there are **fewer parameters involved** (not necessarily positive or negative).  Neural analogies is a little more complex, with as many parameters as the combination of two algorithms.

4. Neural analogies is designed to work with images, and can only support the RGB format for its masks. Semantic style transfer was designed to **integrate with other neural networks** (for pixel labeling and semantic segmentation), and can use any format for its maps, including RGBA or many channels per label masks.

5. Semantic style transfer is **about 25% faster and uses less memory** too.  For neural analogies, the extra computation is effectively the analogy prior — which could improve the quality of the results in theory. In practice, it's hard to tell at this stage and more testing is needed.

If you have any comparisons or insights, be sure to let us know!

----

|Python Version| |License Type| |Project Stars|

.. |Python Version| image:: http://aigamedev.github.io/scikit-neuralnetwork/badge_python.svg
    :target: https://www.python.org/

.. |License Type| image:: https://img.shields.io/badge/license-New%20BSD-blue.svg
    :target: https://github.com/alexjc/neural-doodle/blob/master/LICENSE

.. |Project Stars| image:: https://img.shields.io/github/stars/alexjc/neural-doodle.svg?style=flat
    :target: https://github.com/alexjc/neural-doodle/stargazers
