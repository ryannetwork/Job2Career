package com.intel.analytics.bigdl.apps.job2Career

import com.intel.analytics.bigdl.apps.recommendation.Evaluation
import com.intel.analytics.bigdl.apps.recommendation.Utils._
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}

object TrainWithALS {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val input = "/Users/guoqiong/intelWork/projects/jobs2Career/data/indexed_application_job_resume_2016_2017_10"
    val indexed = spark.read.parquet(input + "/indexed")
    val Array(training, test) = indexed.randomSplit(Array(0.8, 0.2), seed = 1L)

    val toDoubleUdf = udf((num: Float) => num.toDouble)
    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setImplicitPrefs(true)
      .setUserCol("userIdIndex")
      .setItemCol("itemIdIndex")
      .setRatingCol("label")

    val model = als.fit(training)

    val testWithNegative = getNegativeSamples(5, test).union(test)


    val predictions = model.transform(testWithNegative)
      .withColumn("prediction", toDoubleUdf(col("prediction")))
      .withColumn("label", toDoubleUdf(col("label")))

    predictions.printSchema()
    val binaryEva = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
    println("AUROC: " + binaryEva.evaluate(predictions))

    Evaluation.evaluate(predictions)


    val predictions1 = model.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("label")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions1)
    println(s"Root-mean-square error = $rmse")
  }

}
