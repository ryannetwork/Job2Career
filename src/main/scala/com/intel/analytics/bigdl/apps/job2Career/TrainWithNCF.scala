package com.intel.analytics.bigdl.apps.job2Career

import com.intel.analytics.bigdl.apps.recommendation.Utils._
import com.intel.analytics.bigdl.apps.recommendation.{Evaluation, ModelUtils, ModelParam}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.{DLClassifier, DLModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object TrainWithNCF {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("Train")
      .setMaster("local[8]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Engine.init

    val input = "/Users/guoqiong/intelWork/projects/jobs2Career/data/indexed_application_job_resume_2016_2017_10"
    val indexed = spark.read.parquet(input + "/indexed")
    val userCount = spark.read.parquet(input + "/userDict").select("userIdIndex").distinct().count().toInt
    val itemCount = spark.read.parquet(input + "/itemDict").select("itemIdIndex").distinct().count().toInt

    val dataWithNegative = addNegativeSample(5, indexed)
      .withColumn("userIdIndex", add1(col("userIdIndex")))
      .withColumn("itemIdIndex", add1(col("itemIdIndex")))
      .withColumn("label", add1(col("label")))

    val dataInLP = df2LP(dataWithNegative)

    val Array(trainingDF, validationDF) = dataInLP.randomSplit(Array(0.8, 0.2), seed = 1L)

    trainingDF.cache()
    validationDF.cache()

    trainingDF.show(3)

    val time1 = System.nanoTime()
    val modelParam = ModelParam(userEmbed = 20,
      itemEmbed = 20,
      midLayers = Array(40, 20),
      labels = 2)

    val recModel = new ModelUtils(modelParam)

    // val model = recModel.ncf(userCount, itemCount)
    val model = recModel.mlp(userCount, itemCount)

    val criterion = ClassNLLCriterion()
    //val criterion = MSECriterion[Float]()

    val dlc = new DLClassifier(model, criterion, Array(2))
      .setBatchSize(1000)
      .setOptimMethod(new Adam())
      .setLearningRate(1e-2)
      .setLearningRateDecay(1e-5)
      .setMaxEpoch(10)

    val dlModel: DLModel[Float] = dlc.fit(trainingDF)

    println("featuresize " + dlModel.featureSize)
    println("model weights  " + dlModel.model.getParameters())
    val time2 = System.nanoTime()

    val predictions = dlModel.setBatchSize(1).transform(validationDF)

    val time3 = System.nanoTime()

    predictions.cache()
    predictions.show(3)
    predictions.printSchema()


    Evaluation.evaluate(predictions.withColumn("label", toZero(col("label")))
      .withColumn("prediction", toZero(col("prediction"))))

    val time4 = System.nanoTime()

    val trainingTime = (time2 - time1) * (1e-9)
    val predictionTime = (time3 - time2) * (1e-9)
    val evaluationTime = (time4 - time3) * (1e-9)

    println("training time(s):  " + toDecimal(3)(trainingTime))
    println("prediction time(s):  " + toDecimal(3)(predictionTime))
    println("evaluation time(s):  " + toDecimal(3)(predictionTime))
  }

}