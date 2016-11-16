package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._

import org.apache.spark._
import org.apache.spark.rdd.RDD

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.ml.{ Pipeline, PipelineStage }
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import java.lang.Math;

import org.apache.spark.mllib.evaluation.RegressionMetrics

/**
  * Created by efokou and willie on 27/10/16.
  */
object JobML {

  def main(args: Array[String]): Unit = {


    /********************************************************************************
      *
      *        Reading arguments
      *
      ********************************************************************************/

    var dataFileName = ""
    var modelPath = ""

    if(args.length != 2){
      println("Missing Arguments (2 parameters needed): Give the 'path of dataset' and  'path for saving model'")
      println("'path of dataset can be for example /cal/homes/user/Desktop/cleanedDataFrame.parquet")
      println("'path for saving model can be for example /cal/homes/user/Desktop/modelPlanet.model")
      sys.exit(0)
    }else{
      dataFileName = args(0)
      modelPath = args(1)
    }

    /********************************************************************************
      *
      *        SparkSession configuration
      *
      ********************************************************************************/

    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._

    /********************************************************************************
      *
      *        Loading of data (Give path to data from command line). Here we load a Parquet file
      *
      ********************************************************************************/


    val df = spark.read.parquet(dataFileName)        
    // Return the schema of this DataFrame
    df.printSchema()

    /********************************************************************************
      *
      *        Extraction of Features. To build a classifier model, you first extract
      *        the features that most contribute to the classification. For this 	
      *	       dataset we will remove two features: koi_disposition and rowid. One Important 
      *        class will be used:
      *	       VectorAssembler:  used to transform and return a new DataFrame with all of the feature columns in a vector column
      *
      ********************************************************************************/

    // Get list of features
    var featureCols = df.columns

    // define the feature columns to put in the feature vector. Removing of koi_disposition and rowid features
    featureCols = featureCols.filter(x => (!x.contains("koi_disposition") && !x.contains("rowid")))

    //set the input and output column names
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    //return a dataframe with all of the define feature columns in  a vector column. 
    //The transform method produced a new column: features.
    val df1 = assembler.transform(df)

    /********************************************************************************
      *
      *        Creation of a dataframe with the class column "rowid" (“CONFIRMED” or “FALSE-POSITIVE”) 
      *        added as a label . One Important class will be used:
      *	       StringIndexer: used to return a Dataframe with the class column added as a label .
      *
      ********************************************************************************/

    //  Create a label column with the StringIndexer
    val labelIndexer = new StringIndexer().setInputCol("koi_disposition").setOutputCol("label")

    // the  transform method produced a new column: label.
    val df2 = labelIndexer.fit(df1).transform(df1).select("features","label")

    /********************************************************************************
      *
      *        Splitting of data into a training data set and a test data set, 90% of the data is used to 
      *        train the model, and 10% will be used for testing.
      *
      ********************************************************************************/

    var splitSeed = 5043
    val Array(trainingData, testData) = df2.randomSplit(Array(0.9, 0.1), splitSeed)

    // caching of training and testing data in memory. This is useful in practice for iterative algorithm
    trainingData.cache()
    testData.cache()

    /********************************************************************************
      *
      *        Creation of logistic regression Model to train (fit) the model with the training data
      *
      ********************************************************************************/

    // create the classifier,  set parameters for training
    var lr = new LogisticRegression()
      .setElasticNetParam(1.0)  // L1-norm regularization : LASSO
      .setLabelCol("label")
      .setStandardization(true)  // to scale each feature of the model
      .setFitIntercept(true)  // we want an affine regression (with false, it is a linear regression)
      .setTol(1.0e-5)  // stop criterion of the algorithm based on its convergence
      .setMaxIter(300)  // a security stop criterion to avoid infinite loops

    /********************************************************************************
      *
      *        Using of a ParamGridBuilder to construct a grid of parameters to search over.
      *
      ********************************************************************************/

    // We will iterate on log scale for regParam
    val regRange = -6.0 to (0.0, 0.5) toArray
    val regParam = regRange.map(x => math.pow(10,x))

    // Building of grid
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, regParam)
      .build()

    /********************************************************************************
      *
      *        Building BinaryClasssificationEvaluator
      *
      ********************************************************************************/

    // create an Evaluator for binary classification
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")

    /********************************************************************************
      *
      *        Using TrainValidationSplit for hyper-parameter tuning
      *
      ********************************************************************************/

    // 70% of the data will be used for training and the remaining 20% for validation.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    /********************************************************************************
      *
      *        Training of model
      *
      ********************************************************************************/

    // Run trainValidationSplit, and choose the best set of parameters.
    val model = trainValidationSplit.fit(trainingData)

    /********************************************************************************
      *
      *        Testing of model and printing of results
      *
      ********************************************************************************/

    // run the  model on test features to get predictions
    val df_WithPredictions = model.transform(testData)
    df_WithPredictions.groupBy("label", "prediction").count.show()

    evaluator.setRawPredictionCol("prediction")

    // best model parameters
    println("\n best model parameters : \n")
    print(model.bestModel.extractParamMap().toString())

    // others performances values
    println("\n\n Other performances values : \n")
    val accuracy = evaluator.evaluate(df_WithPredictions)
    println("Model accuracy : "  + accuracy.toString())

    val lp = df_WithPredictions.select( "label", "prediction")
    val counttotal = df_WithPredictions.count()
    println("counttotal : "  + counttotal.toString())

    val correct = lp.filter($"label" === $"prediction").count()
    println("correct : "  + correct.toString())

    val wrong = lp.filter(not($"label" === $"prediction")).count()
    println("wrong : "  + wrong.toString())

    val truep = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
    println("truep : "  + truep.toString())

    val falseN = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count()
    println("falseN : "  + falseN.toString())

    val falseP = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count()
    println("falseP : "  + falseP.toString())

    val ratioWrong=wrong.toDouble/counttotal.toDouble
    println("ratioWrong : "  + ratioWrong.toString())

    val ratioCorrect=correct.toDouble/counttotal.toDouble
    println("ratioCorrect : "  + ratioCorrect.toString())

    val rm = new RegressionMetrics(
      df_WithPredictions.select("prediction", "label").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    )
    println("\nMSE: " + rm.meanSquaredError)
    println("MAE: " + rm.meanAbsoluteError)
    println("RMSE Squared: " + rm.rootMeanSquaredError)
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")



    /********************************************************************************
      *
      *        Saving the model
      *
      ********************************************************************************/

    model.write.overwrite.save(modelPath)



  }
}
