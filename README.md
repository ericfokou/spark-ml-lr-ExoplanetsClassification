# spark-ml-lr-ExoplanetsClassification
Build a classification model of exoplanets labeled "confirmed" or "false-positive". we will use Spark Machine Learning Logistic regression written in Scala for building our model.

============
Team
============


* [Eric FOKOU](https://github.com/ericfokou/)
* [Willie DROUHET](https://github.com/drwi)
   
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
<p align="left">
  <img src="https://raw.githubusercontent.com/ericfokou/spark-ml-lr-ExoplanetsClassification/master/media/Satellite_observation.png" alt="Luminosity curve" height="300" width="450""/>
</p>
As there are about 200 billion stars in our galaxy, and therefore potentially as much (or even more!) Exoplanets, their detection must be automated to "scale up". The method of transits is already automatic (more than 22 million curves of luminosity recorded by Kepler), but not the confirmation of the candidate planets, hence the automatic classifier that we will build.

============
Data
============

The data on exoplanets are public and accessible [link] (http://exoplanetarchive.ipac.caltech.edu/index.html). There are already 3388 confirmed exoplanets and about as many false positives, our classifier will be trained on these data. There is one exoplanet per line. The column of labels (what we are going to try to predict) is called "koi_disposition". You can retrieve the already cleaned dataset in parquet format in the project directly (cleanedDataFrame.parquet). The classifier will only use information from the brightness curves.

============
Installation
============

Download the project in a local directory

```
git clone https://github.com/ericfokou/spark-ml-lr-ExoplanetsClassification.git
```

Go into the imported directory (spark-ml-lr-ExoplanetsClassification) and run following command:

```
sbt assembly
```

N.B. : The generated file is located in spark-ml-lr-ExoplanetsClassification/target/scala-2.11/spark-ml-lr-ExoplanetsClassification-assembly-1.0.jar

============
Run
============

Go to bin local directory of spark:

```
cd spark-2.0.0-bin-hadoop2.6/bin
```

Submit the script. Give three parameters:

* First: absolute path of the .jar (/homes/efokou/spark-ml-lr-ExoplanetsClassification-1.0.jar, for example)
* Second: dataset file path, ie absolute path of the .parquet copied from git repo (/homes/efokou/cleanedDataFrame.parquet, for example)
* Third: absolute path of the model that will be saved (/homes/efokou/modelPlanet.model, for example)

The script submission should go along the lines of...

```
./spark-submit --conf spark.eventLog.enabled=true --conf spark.eventLog.dir="/tmp" --driver-memory 3G --executor-memory 4G --class com.sparkProject.JobML  [absolute path of the .jar] [absolute path of the .parquet copied from the git repo] [absolute path of the model that will be saved]
```

For example the authors used the following command for running the job:

```
./spark-submit --conf spark.eventLog.enabled=true --conf spark.eventLog.dir="/tmp" --driver-memory 3G --executor-memory 4G --class com.sparkProject.JobML /homes/efokou/spark-ml-lr-ExoplanetsClassification-1.0.jar /homes/efokou/cleanedDataFrame.parquet /homes/efokou/modelPlanet.model
```
============
Output
============
<p align="center">
  <img src="https://raw.githubusercontent.com/ericfokou/spark-ml-lr-ExoplanetsClassification/master/media/output.png" alt="Luminosity curve""/>
</p>



