package com.intel.analytics.zoo.apps.job2Career

import com.intel.analytics.zoo.apps.recommendation.Utils._
import com.intel.analytics.zoo.apps.job2Career.TrainWithD2VGlove._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.apps.recommendation.{Evaluation, ModelParam, ModelUtils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext, SaveMode, SparkSession}
import scopt.OptionParser
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.nnframes.{NNClassifier, NNClassifierModel, NNEstimator, NNModel}

case class TrainParam(val inputDir: String = "/Users/guoqiong/intelWork/projects/jobs2Career/data/indexed_application_job_resume_2016_2017_10",
                      val outputDir: String = "/Users/guoqiong/intelWork/projects/jobs2Career/data/validation_predict",
                      val topK: Int = 500,
                      val dictDir: String = "/Users/guoqiong/intelWork/projects/wrapup/textClassification/keras/glove.6B/glove.6B.50d.txt",
                      val valDir: String = "/Users/guoqiong/intelWork/projects/jobs2Career/data/validation/part*",
                      val batchSize: Int = 8000,
                      val nEpochs: Int = 5,
                      val learningRate: Double = 0.005,
                      val learningRateDecay: Double = 1e-6)

object TrainWithNCF_Glove {

  def main(args: Array[String]): Unit = {

    val defaultParams = TrainParam()

    val conf = new SparkConf()
    conf.setAppName("NCFExample").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.getNNContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)
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

    val splitNum = (userDict.count() * 0.8).toInt

    //    val indexedTrain = indexed.filter(col("userIdIndex") <= splitNum)
    //    val indexedValidation = indexed.filter(col("userIdIndex") > splitNum)

    val indexedWithNegative = DataProcess.negativeJoin(indexed, itemDict, userDict, negativeK = 1)
      .withColumn("label", add1(col("label")))

    val Array(trainWithNegative, validationWithNegative) = indexedWithNegative.randomSplit(Array(0.8, 0.2), 1L)

    println("---------distribution of label trainWithNegative ----------------")
    // trainWithNegative.select("label").groupBy("label").count().show()
    val trainingDF = getFeaturesLP(trainWithNegative)
    val validationDF = getFeaturesLP(validationWithNegative)

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

    val criterion = ClassNLLCriterion[Float]()
    val dlc =  NNClassifier(model, criterion, Array(100))
      .setBatchSize(param.batchSize)
      .setOptimMethod(new Adam())
      .setLearningRate(param.learningRate)
      .setLearningRateDecay(param.learningRateDecay)
      .setMaxEpoch(param.nEpochs)

    //    val dlc: DLEstimator[Float] = new DLClassifier(model, criterion, Array(100))
    //      .setBatchSize(param.batchSize)
    //      .setOptimMethod(new Adam())
    //      .setLearningRate(param.learningRate)
    //      .setLearningRateDecay(param.learningRateDecay)
    //      .setMaxEpoch(param.nEpochs)

    val nNModel = dlc.fit(trainingDF)

    println(nNModel.model.getParametersTable())
    nNModel.model.saveModule(modelPath, null, true)

    val time2 = System.nanoTime()

    val predictions: DataFrame = nNModel.transform(validationDF)

    val time3 = System.nanoTime()

    predictions.cache().count()
    predictions.show(20)
    println("validation results")
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
    val nNModel =  NNClassifierModel(loadedModel, Array(100))
      .setBatchSize(para.batchSize)

    val validationIn = spark.read.parquet(para.valDir)
    // validationIn.printSchema()
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
    val validationLP = getFeaturesLP(validationVectors).coalesce(32)

    val predictions2: DataFrame = nNModel.transform(validationLP)

    predictions2.persist().count()
    predictions2.select("userId", "itemId", "label", "prediction").show(20, false)

    println("validation results on golden dataset")
    val dataToValidation = predictions2.withColumn("label", toZero(col("label")))
      .withColumn("prediction", toZero(col("prediction")))
    Evaluation.evaluate2(dataToValidation)

    predictions2.write.mode(SaveMode.Overwrite).parquet(para.outputDir)

  }

}