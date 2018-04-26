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

    val defaultParams = TrainParam()

    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("app")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    //spark.sparkContext.setLogLevel("WARN")

    val parser = new OptionParser[TrainParam]("BigDL Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[String]("outputDir")
        .text(s"outputDir")
        .action((x, c) => c.copy(outputDir = x))
      opt[String]("dictDir")
        .text(s"wordVec data")
        .action((x, c) => c.copy(dictDir = x))
      opt[String]("valDir")
        .text(s"valDir data")
        .action((x, c) => c.copy(valDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Double]('l', "learningRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x.toDouble))
      opt[Double]('d', "learningRateDecay")
        .text("learning rate decay")
        .action((x, c) => c.copy(learningRateDecay = x.toDouble))

    }

    parser.parse(args, defaultParams).map {
      params =>
        run(spark: SparkSession, params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(spark: SparkSession, param: TrainParam): Unit = {

    Engine.init
    val input = param.inputDir
    val modelPath = param.inputDir + "/model"

    val indexed = spark.read.parquet(input + "/indexed")
    val userDict = spark.read.parquet(input + "/userDict")
    val itemDict = spark.read.parquet(input + "/itemDict")

    val Array(indexedTrain, indexedValidation) = indexed.randomSplit(Array(0.8, 0.2), seed = 1L)
    val trainWithNegative = DataProcess.negativeJoin(indexedTrain, itemDict, userDict, negativeK = 1)
      .withColumn("label", add1(col("label")))
    val validationWithNegative = DataProcess.negativeJoin(indexedValidation, itemDict, userDict, negativeK = 1)
      .withColumn("label", add1(col("label")))

    println("---------distribution of label trainWithNegative ----------------")
    // trainWithNegative.select("label").groupBy("label").count().show()
    val trainingDF = getFeaturesLP(trainWithNegative)
    val validationDF = getFeaturesLP(validationWithNegative)

    trainingDF.printSchema()
    trainingDF.groupBy("label").count().show()
    validationDF.groupBy("label").count().show()

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

    val dlc: DLEstimator[Float] = new DLClassifier(model, criterion, Array(100))
      .setBatchSize(param.batchSize)
      .setOptimMethod(new Adam())
      .setLearningRate(param.learningRate)
      .setLearningRateDecay(param.learningRateDecay)
      .setMaxEpoch(param.nEpochs)

    val dlModel: DLModel[Float] = dlc.fit(trainingDF)

    dlModel.model.saveModule(modelPath, null, true)

    val time2 = System.nanoTime()

    val predictions: DataFrame = dlModel.transform(validationDF)

    val time3 = System.nanoTime()

    predictions.cache()
    predictions.show(30)
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

  def processGoldendata(spark: SparkSession, para: TrainParam, modelPath: String) = {

    val loadedModel = Module.loadModule(modelPath, null)
    val dlModel = new DLClassifierModel[Float](loadedModel, Array(100))

    val validationIn = spark.read.parquet(para.valDir)
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

    predictions2.coalesce(8).write.mode(SaveMode.Overwrite).parquet(para.outputDir)

  }

}