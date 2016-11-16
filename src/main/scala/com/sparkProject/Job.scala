package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._

object Job {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._


    /********************************************************************************
      *
      *        TP 1
      *
      *        - Set environment, InteliJ, submit jobs to Spark
      *        - Load local unstructured data
      *        - Word count , Map Reduce
      ********************************************************************************/




    // ----------------- word count ------------------------

    val df_wordCount = sc.textFile("/cal/homes/efokou/INF729/README.md")
      .flatMap{case (line: String) => line.split(" ")}
      .map{case (word: String) => (word, 1)}
      .reduceByKey{case (i: Int, j: Int) => i + j}
      .toDF("word", "count")

    df_wordCount.orderBy($"count".desc).show()




    /********************************************************************************
      *
      *        TP 2 : début du projet
      *
      ********************************************************************************/

    val df = spark
      .read // returns a DataFrameReader, giving access to methods “options” and “csv”
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .option("comment", "#") // All lines starting with # are ignored
      .csv("/cal/homes/efokou/INF729/cumulative.csv")

    println("number of columns", df.columns.length) //df.columns returns an Array of columns names, and arrays have a method “length” returning their length
    println("number of rows", df.count)

    val columns = df.columns.slice(10, 20) // df.columns returns an Array. In scala arrays have a method “slice” returning a slice of the array
    df.select(columns.map(col): _*).show(50) //
    //df.select(df.columns.slice(10,20) : _*).show(50) // moi
    df.show()

    df.printSchema()
    //df.dtypes

    df.groupBy($"koi_disposition").count().show()

    val df_cleaned =  df.filter($"koi_disposition" === "CONFIRMED" || $"koi_disposition" === "FALSE POSITIVE")
    df_cleaned.show()

    df_cleaned.groupBy($"koi_eccen_err1").count().show()
    //df_cleaned.select($"koi_eccen_err1").distinct.count

    val df_cleaned2 = df_cleaned.drop($"koi_eccen_err1")

    val df_cleaned3 = df_cleaned2.drop("index","kepid","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec",
      "koi_sparprov","koi_trans_mod","koi_datalink_dvr","koi_datalink_dvs","koi_tce_delivname",
      "koi_parm_prov","koi_limbdark_mod","koi_fittype","koi_disp_prov","koi_comment","kepoi_name","kepler_name",
      "koi_vet_date","koi_pdisposition")

    df_cleaned3.show()

    //val df_clean4 = df_cleaned3.columns.map(col => if(df_cleaned3.select(col).distinct().count() == 1) { df_cleaned3.drop(col) })

    val useless_column = df_cleaned3.columns.filter{ case (column:String) =>
      df_cleaned3.agg(countDistinct(column)).first().getLong(0) <= 1 }


    val df_cleaned4 = df_cleaned3.drop(useless_column: _*)
    //df_cleaned4.describe("koi_impact", "koi_duration").show()

    val df_clean5 = df_cleaned4.na.fill(0.0)
    //val df_clean5 = df_clean4.na.fill(0,Seq("blank"))

    val df_labels = df_clean5.select("rowid", "koi_disposition")
    val df_features = df_clean5.drop("koi_disposition")
    val df_joined = df_features.join(df_labels, usingColumn = "rowid")
    //val df_joined = df_features.join(df_labels, df_features("rowid") === df_labels("rowid"))

    def udf_sum = udf((col1: Double, col2: Double) => col1 + col2)

    val df_newFeatures = df_joined.withColumn("koi_ror_min", udf_sum($"koi_ror", $"koi_ror_err2"))
            .withColumn("koi_ror_max", $"koi_ror" + $"koi_ror_err1")

    df_newFeatures.coalesce(1) // optional : regroup all data in ONE partition, so that results are printed in ONE file
      // >>>> You should not do that in general, only when the data are small enough to fit in the memory of a single machine.
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv("/cal/homes/efokou/INF729/cumulative_cleaned.csv")

    println("number of rows", df.count)

  }
}