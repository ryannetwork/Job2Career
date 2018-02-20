package com.intel.analytics.bigdl.apps.job2Career

import com.intel.analytics.bigdl.apps.recommendation.Utils._
import com.intel.analytics.bigdl.apps.recommendation.{Evaluation, ModelParam, ModelUtils}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.{DLClassifier, DLModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}
import scopt.OptionParser

object TrainWithNCF_Glove {

  def main(args: Array[String]): Unit = {

    val defaultParams = DataParams()

    val parser = new OptionParser[DataParams]("BigDL Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[String]("outputDir")
        .text(s"outputDir")
        .action((x, c) => c.copy(outputDir = x))
      opt[String]("dictDir")
        .text(s"wordVec data")
        .action((x, c) => c.copy(dictDir = x))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: DataParams): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("app")
      .setMaster("local[8]")
    val sc = new SparkContext(conf)

    val spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    Engine.init

    val input = param.inputDir
    val indexed = spark.read.parquet(input + "/indexed")
    val userDict = spark.read.parquet(input + "/userDict")
    val itemDict = spark.read.parquet(input + "/itemDict")

    val dataWithNegative = DataProcess.negativeJoin(indexed, itemDict, userDict, negativeK = 5)
      .withColumn("label", add1(col("label")))
      .select("userVec", "itemVec", "label")

    dataWithNegative.show(10, false)
    val dataInLP = df2LP2(dataWithNegative)

    val Array(trainingDF, validationDF) = dataInLP.randomSplit(Array(0.8, 0.2), seed = 1L)

    trainingDF.cache()
    validationDF.cache()

    trainingDF.show(3, false)

    val time1 = System.nanoTime()
    val modelParam = ModelParam(userEmbed = 20,
      itemEmbed = 20,
      midLayers = Array(40, 20),
      labels = 2)

    val recModel = new ModelUtils(modelParam)

    // val model = recModel.ncf(userCount, itemCount)
    val model = recModel.mlp2

    val criterion = ClassNLLCriterion()
    // val criterion = MSECriterion[Float]()

    val dlc = new DLClassifier(model, criterion, Array(100))
      .setBatchSize(1000)
      .setOptimMethod(new Adam())
      .setLearningRate(1e-2)
      .setLearningRateDecay(1e-5)
      .setMaxEpoch(10)

    val dlModel: DLModel[Float] = dlc.fit(trainingDF)

    println("featuresize " + dlModel.featureSize)
    println("model weights  " + dlModel.model.getParameters())
    val time2 = System.nanoTime()

    val predictions = dlModel.setBatchSize(10).transform(validationDF)

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