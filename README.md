# spark-ml-lr-ExoplanetsClassification
Build a classification model of exoplanets labeled "confirmed" or "false-positive". we will use Spark Machine Learning Logistic regression written in Scala for building our model.
   
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

============
Data
============

The data on exoplanets are public and accessible [link] (http://exoplanetarchive.ipac.caltech.edu/index.html). There are already 3388 confirmed exoplanets and about as many false positives, our classifier will be trained on these data. There is one exoplanet per line. The column of labels (what we are going to try to predict) is called "koi_disposition". You can retrieve the already cleaned dataset in parquet format in the project directly (cleanedDataFrame.parquet). The classifier will only use information from the brightness curves.

============
Install
============

Download the project, then unpack it. Import it into IntelliJ; run teh following command:
* In the homepage click on "import project".
* Select the path to the decompressed project.
* Select "import project from external model", and select SBT => next
* Select "use auto import" => finish
* SBT project data to import: check that both folders are selected => OK
* Waiting for the project to be loaded and its dependencies

============
Run
============

Compile and create a jar file

```
sbt assembly
```

In a terminal go where the spark-submit is:

```
cd spark-2.0.0-bin-hadoop2.6/bin
```

Submit the script. Give two parameters:

* First: dataset file path
* Model file path to save

This is one example of 

```
./spark-submit --conf spark.eventLog.enabled=true --conf spark.eventLog.dir="/tmp" --driver-memory 3G --executor-memory 4G --class com.sparkProject.JobML /cal/homes/efokou/INF729/tp_spark/target/scala-2.11/tp_spark-assembly-1.0.jar /homes/efokou/cleanedDataFrame.parquet /homes/efokou/modelPlanet.model
```

============
Contributors
============

