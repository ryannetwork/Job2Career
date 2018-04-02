package com.intel.analytics.bigdl.apps.job2Career

import com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove.loadWordVecMap
import com.intel.analytics.bigdl.apps.recommendation.Utils._
import com.intel.analytics.bigdl.apps.recommendation.{Evaluation, ModelParam, ModelUtils}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.{DLClassifier, DLClassifierModel, DLEstimator, DLModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
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
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Double]('l', "lRate")
        .text("learning rate")
        .action((x, c) => c.copy(lRate = x.toDouble))

    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: DataParams): Unit = {
    println("learning rate: " + param.lRate)

    //Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("app")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    //spark.sparkContext.setLogLevel("WARN")
    Engine.init

    val input = param.inputDir
    val modelPath = param.inputDir + "/model"

    val indexed = spark.read.parquet(input + "/indexed")
    val userDict = spark.read.parquet(input + "/userDict")
    val itemDict = spark.read.parquet(input + "/itemDict")

    val dataWithNegative = DataProcess.negativeJoin(indexed, itemDict, userDict, negativeK = 5)
      .withColumn("label", add1(col("label")))
      .select("userVec", "itemVec", "label")

    dataWithNegative.show(2)
    val dataInLP = getFeaturesLP(dataWithNegative)


    dataInLP.printSchema()
    val Array(trainingDF, validationDF) = dataInLP.randomSplit(Array(0.8, 0.2), seed = 1L)

    println("training data-----------")
    trainingDF.show(2)

    trainingDF.cache()
    validationDF.cache()

    val time1 = System.nanoTime()
    val modelParam = ModelParam(userEmbed = 20,
      itemEmbed = 20,
      midLayers = Array(40, 20),
      labels = 2)

    val recModel = new ModelUtils(modelParam)

    // val model = recModel.ncf(userCount, itemCount)
    val model = recModel.mlp3

    val criterion = ClassNLLCriterion()
    //  val criterion = MSECriterion[Float]()

    val dlc: DLEstimator[Float] = new DLClassifier(model, criterion, Array(100))
      .setBatchSize(param.batchSize)
      .setOptimMethod(new Adam())
      .setLearningRate(param.lRate)
      .setLearningRateDecay(1e-5)
      .setMaxEpoch(param.nEpochs)

    val dlModel: DLModel[Float] = dlc.fit(trainingDF)

    dlModel.model.saveModule(modelPath, true)

    val time2 = System.nanoTime()

    val predictions: DataFrame = dlModel.transform(validationDF)

    val time3 = System.nanoTime()

    predictions.cache()
    predictions.show(3)
    predictions.printSchema()

    Evaluation.evaluate2(predictions.withColumn("label", toZero(col("label")))
      .withColumn("prediction", toZero(col("prediction"))))

    val time4 = System.nanoTime()

    val trainingTime = (time2 - time1) * (1e-9)
    val predictionTime = (time3 - time2) * (1e-9)
    val evaluationTime = (time4 - time3) * (1e-9)

    println("training time(s):  " + toDecimal(3)(trainingTime))
    println("prediction time(s):  " + toDecimal(3)(predictionTime))
    println("evaluation time(s):  " + toDecimal(3)(predictionTime))

    processGoldendata(spark, param, modelPath)
    println("stop")

  }


  def processGoldendata(spark: SparkSession, para: DataParams, modelPath: String) = {

    val loadedModel = Module.loadModule(modelPath)
    val dlModel = new DLClassifierModel[Float](loadedModel, Array(100))

    val input = "/Users/guoqiong/intelWork/projects/jobs2Career/data/validation/part*"
    val validationIn = spark.read.parquet(input)
    validationIn.printSchema()
    val validationDF = validationIn
      .select("resume_id", "job_id", "resume.resume.normalizedBody", "description", "apply_flag")
      .withColumn("label", add1(col("apply_flag")))
      .withColumnRenamed("resume.resume.normalizedBody", "normalizedBody")
      .withColumnRenamed("resume_id", "userId")
      .withColumnRenamed("normalizedBody", "userDoc")
      .withColumnRenamed("description", "itemDoc")
      .withColumnRenamed("job_id", "itemId")
      .select("userId", "itemId", "userDoc", "itemDoc", "label")
      .filter(col("itemDoc").isNotNull && col("userDoc").isNotNull && col("userId").isNotNull
        && col("itemId").isNotNull && col("label").isNotNull)

    val dict: Map[String, Array[Float]] = loadWordVecMap(para.dictDir)
    val br: Broadcast[Map[String, Array[Float]]] = spark.sparkContext.broadcast(dict)

    val validationCleaned = DataProcess.cleanData(validationDF, br.value)
    val validationVectors = DataProcess.getGloveVectors(validationCleaned, br)
    val validationLP = getFeaturesLP(validationVectors)

    val predictions2: DataFrame = dlModel.transform(validationLP)

    predictions2.persist()
    predictions2.select("userId", "itemId", "label", "prediction").show(50, false)

    val dataToValidation = predictions2.withColumn("label", toZero(col("label")))
      .withColumn("prediction", toZero(col("prediction")))
    Evaluation.evaluate2(dataToValidation)

    // predictions2.coalesce(8).write.mode(SaveMode.Overwrite).parquet("/Users/guoqiong/intelWork/projects/jobs2Career/data/validation_predict")

  }

}