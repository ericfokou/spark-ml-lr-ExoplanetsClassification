# spark-ml-lr-ExoplanetsClassification
Build a classification model of exoplanets labeled "confirmed" or "false-positive". we will use Spark Machine Learning Logistic regression written in Scala for building our model.


.. contents::

.. section-numbering::

.. raw:: pdf

   PageBreak oneColumn
   
=============
Setup 
=============

* Spark 2.0.0
* Java 1.8
* SBT
* IntelliJ Community

============
Context
============

Exoplanets are planets rotating around other stars than the Sun. Their study allows us to better understand how the solar system was formed, and a fraction of them could be conducive to the development of extraterrestrial life! They are detected in two steps:
* A satellite (Kepler) observes the stars and locates those whose luminosity curve presents a "hollow", which could indicate that a planet has passed
(Part of the light emitted by the star being obscured by the passage of the planet). This method of "transit" allows us to define candidate exoplanets, and to deduce the characteristics that the planet would have if it really existed (distance to its star, diameter, shape of its orbit, etc.).
* It is then necessary to validate or invalidate the candidates using another more expensive method, based on measurements of radial velocities of the star. Candidates are then classified as "confirmed" or "false-positive".
<p align="center">
  <img src="https://raw.githubusercontent.com/ericfokou/spark-ml-lr-ExoplanetsClassification/master/media/Satellite_observation.png" alt="Luminosity curve" height="300" width="450""/>
</p>
As there are about 200 billion stars in our galaxy, and therefore potentially as much (or even more!) Exoplanets, their detection must be automated to "scale up". The method of transits is already automatic (more than 22 million curves of luminosity recorded by Kepler), but not the confirmation of the candidate planets, hence the automatic classifier that we will build.