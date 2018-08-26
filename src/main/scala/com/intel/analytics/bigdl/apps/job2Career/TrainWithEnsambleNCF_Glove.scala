package com.intel.analytics.bigdl.apps.job2Career

import com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove.loadWordVecMap
import com.intel.analytics.bigdl.apps.job2Career.Utils.{AppParams, KmeansParam, NCFParam}
import com.intel.analytics.bigdl.apps.recommendation.Evaluation.evaluatePredictions
import com.intel.analytics.bigdl.apps.recommendation.Utils._
import com.intel.analytics.bigdl.apps.recommendation.{Evaluation, ModelParam, ModelUtils, NCFJob2Career}
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.common.ZooModel
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, _}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.storage.StorageLevel

object TrainWithEnsambleNCF_Glove {

  val logger = Logger.getLogger(getClass)
  Logger.getLogger("org").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
    conf.setAppName("jobs2career").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)
    Utils.trainParser.parse(args, Utils.AppParams()).map { param =>
      run(sqlContext: SQLContext, param)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(sqlContext: SQLContext, param: AppParams): Unit = {

    def readPreprocessed(): (DataFrame, DataFrame, DataFrame) = {
      val input: String = param.dataPathParams.preprocessedDir
      val indexed = sqlContext.read.parquet(input + "/indexed").drop("itemIdOrg").drop("userIdOrg")
      val userDict = sqlContext.read.parquet(input + "/userDict")
      val itemDict = sqlContext.read.parquet(input + "/itemDict")

    //  val indexedAll = DataProcess.negativeJoin(indexed, itemDict, userDict, param.negativeK)
     //   .withColumn("label", add1(col("label")))
     // indexedAll.write.mode(SaveMode.Overwrite).parquet(param.dataPathParams.preprocessedDir + "/indexedNeg" + param.negativeK)
      (indexed, userDict, itemDict)
    }

    def processNcfWithKmeans(indexed: DataFrame, userDict: DataFrame) = {

      val kmeanUserDict = processKmeans(userDict, param.kmeansParams).persist()
      kmeanUserDict.write.mode(SaveMode.Overwrite).parquet(param.dataPathParams.modelOutput + "/kmeans")
      // val kmeanUserDict = sqlContext.read.parquet(param.dataPathParams.modelOutput + "/kmeans")

      (0 to param.kmeansParams.kClusters - 1).map(kth => {

        val userDictKth = kmeanUserDict.filter(s"kth = $kth")

        val indexedKth = indexed.join(userDictKth.select("userId"), Seq("userId")).persist(StorageLevel.DISK_ONLY)

        println(kth + " cluster data distribution: ")
        indexedKth.groupBy("label").count().show()

        processNcf(kth, indexedKth)
        indexedKth.unpersist()
      })
      kmeanUserDict.unpersist()
    }

    def processNcf(kth: Int, indexed: DataFrame) = {

      val Array(trainDF, validationDF) = indexed.randomSplit(Array(0.8, 0.2), seed = 1L)
      val trainpairFeatureRdds = assemblyFeature(trainDF)

      val trainRdds: RDD[Sample[Float]] = trainpairFeatureRdds.map(x => x.sample)

      val time1 = System.nanoTime()
      val ncfPath = if (kth >= 0) param.ncfParams.modelPath + kth else param.ncfParams.modelPath
      val model = if (param.ncfParams.isTrain) {
        trainNcf(param.ncfParams, trainRdds, ncfPath)
      } else {
        val loadedModel = ZooModel.loadModel(ncfPath, null).asInstanceOf[NCFJob2Career]
        println(loadedModel.getParameters())
        loadedModel
      }

      val predictions = ncfPredict(model, sqlContext, validationDF)
      val metrics = evaluatePredictions(predictions.join(validationDF, Array("userId", "itemId")))
      logger.info("evaluation for validation dataset")
      logger.info("hyperparameters: " + param)
      logger.info(metrics.mkString("|"))
    }

    val mode = param.mode
    mode match {
      case Utils.Mode_NCFWithKeans =>
        val (indexedAll, userDict, itemDict) = readPreprocessed()
        processNcfWithKmeans(indexedAll, userDict)
      case Utils.Mode_NCF =>
        val (indexedAll, userDict, itemDict) = readPreprocessed()
        processNcf(-1, indexedAll)
      case Utils.Mode_Data =>
        DataProcess.preprocess(sqlContext, param)
      case _ =>
        throw new IllegalArgumentException(s"mode $mode not supported")
    }

    evaluateWithGoldenData(sqlContext, param)
  }

  def processKmeans(dataset: DataFrame,
                    param: KmeansParam): DataFrame = {

    val vectorCol = param.featureVec
    val dataml = dataset.withColumn("kmeansFeatures", array2vec(col(vectorCol)))
    val kmeans = new KMeans()
      .setK(param.kClusters)
      .setSeed(1L)
      .setInitMode("k-means||")
      .setMaxIter(param.numIterations)
      .setFeaturesCol("kmeansFeatures")
      .setPredictionCol("kth")

    val kmeanPath = param.modelPath
    val model: KMeansModel = if (param.isTrain) {
      require(kmeanPath != null, "model path can't be null")
      require(dataml.schema.fieldNames.contains(vectorCol), vectorCol + " is needed to train a kmeans model")

      val trained: KMeansModel = kmeans.fit(dataml)
      deleteFile(kmeanPath)
      trained.save(kmeanPath)
      trained
    } else {
      require(kmeanPath != null, "model path can't be null")
      KMeansModel.load(kmeanPath)
    }

    val predictions = model.transform(dataml)
    val evaluator = new ClusteringEvaluator()
      .setFeaturesCol("kmeansFeatures").setPredictionCol("kth")

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    predictions.drop("kmeansFeatures")
  }

  def trainNcf(param: NCFParam, trainRdds: RDD[Sample[Float]], modelPath: String) = {

    val modelMlp = NCFJob2Career(100, 100,
      numClasses = 2,
      userEmbed = 50,
      itemEmbed = 50,
      hiddenLayers = Array(40, 20, 10),
      includeMF = false,
      trainEmbed = false)

    val optimizer = Optimizer(
      model = modelMlp,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = param.batchSize)

    val optimMethod = new Adam[Float](
      learningRate = param.learningRate,
      learningRateDecay = param.learningRateDecay)

    optimizer
      .setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxEpoch(param.nEpochs))
      .optimize()

    println(modelMlp.getParameters())
    modelMlp.saveModel(modelPath, null, true)

    modelMlp
  }

  def ncfPredict(model: NCFJob2Career,
                 sqlContext: SQLContext,
                 validationDF: DataFrame) = {

    val validationpairFeatureRdds = assemblyFeature(validationDF)
    val pairPredictions = model.predictUserItemPair(validationpairFeatureRdds)
    val predictions = sqlContext.createDataFrame(pairPredictions).toDF()

    predictions
  }

  def ncfWithKmeansPredict(param: AppParams, sqlContext: SQLContext, validationDF: DataFrame) = {
    val kmeansModel = KMeansModel.load(param.kmeansParams.modelPath)
    val vectorCol = param.kmeansParams.featureVec
    val validationml = validationDF
      .withColumn("kmeansFeatures", array2vec(col(vectorCol)))

    val validationDFWithK = kmeansModel.transform(validationml).drop("kmeansFeatures")

    val predictionsAll: DataFrame = (0 to param.kmeansParams.kClusters - 1).map(kth => {

      val validationDFKth = validationDFWithK.filter(s"kth = $kth")
      val ncfPath = if (kth >= 0) param.ncfParams.modelPath + kth else param.ncfParams.modelPath
      val ncfModelKth = ZooModel.loadModel(ncfPath, null).asInstanceOf[NCFJob2Career]
      val predictionsKth = ncfPredict(ncfModelKth, sqlContext, validationDFKth)
      predictionsKth
    }).reduceLeft((a, b) => a.union(b))

    predictionsAll.join(validationDFWithK, Array("userId", "itemId"))
  }

  def getGoldenDF(sqlContext: SQLContext, dictDir:String, goldDir:String, labelCol:String = "is_clicked") = {
    val goldIn = sqlContext.read.parquet(goldDir)

   // println("raw data")
   // println(goldIn.count())
    val goldCleaned1 = goldIn
      .select("resume_id", "job_id", "resume.resume.normalizedBody", "description", "apply_flag","is_clicked")
      .withColumn("label", add1(col(labelCol)))
      .withColumnRenamed("resume.resume.normalizedBody", "normalizedBody")
      .withColumnRenamed("normalizedBody", "userDoc")
      .withColumnRenamed("description", "itemDoc")
      .select(col("resume_id"), col("job_id"), col("userDoc"), col("itemDoc"), col("label"))
      .filter(col("itemDoc").isNotNull && col("userDoc").isNotNull && col("resume_id").isNotNull
        && col("job_id").isNotNull && col("label").isNotNull)
      .withColumnRenamed("job_id", "itemId")
      .withColumnRenamed("resume_id", "userId")

    val dict: Map[String, Array[Float]] = loadWordVecMap(dictDir)
    val br: Broadcast[Map[String, Array[Float]]] = sqlContext.sparkContext.broadcast(dict)

    val goldCleaned = DataProcess.cleanData(goldCleaned1, br.value)
      .withColumnRenamed("itemId", "job_id")
      .withColumnRenamed("userId", "resume_id")

   // println("cleaned")
   // println(goldCleaned.count())

    val goldVec = DataProcess.getGloveVectors(goldCleaned, br)

    val si1 = new StringIndexer().setInputCol("resume_id").setOutputCol("userId")
    val si2 = new StringIndexer().setInputCol("job_id").setOutputCol("itemId")

    val pipeline = new Pipeline().setStages(Array(si1, si2))
    val pipelineModel = pipeline.fit(goldVec)
    val goldDF: DataFrame = pipelineModel.transform(goldVec)

//    println("indexed")
 //   println(goldDF.count())
    goldDF
  }

  def evaluateWithGoldenData(sqlContext: SQLContext, param: AppParams) = {

    val validationDF: DataFrame = getGoldenDF(sqlContext, param.gloveParams.dictDir, param.dataPathParams.evaluateDir)

    validationDF.persist(StorageLevel.DISK_ONLY)
    val mode = param.mode
    val predictions: DataFrame = mode match {
      case Utils.Mode_NCFWithKeans => {
        ncfWithKmeansPredict(param, sqlContext, validationDF)
      }
      case Utils.Mode_NCF =>
        val loadedModel = ZooModel.loadModel(param.ncfParams.modelPath, null).asInstanceOf[NCFJob2Career]
        val tmpPredictions = ncfPredict(loadedModel, sqlContext, validationDF)
       (tmpPredictions.join(validationDF, Array("userId", "itemId")))

      case _ =>
        throw new IllegalArgumentException(s"mode $mode not supported")
    }

    val metrics = evaluatePredictions(predictions)
    validationDF.unpersist()
    logger.info("evaluation for golden dataset")
    logger.info("hyperparameters: " + param)
    logger.info(metrics.mkString("|"))
    //    predictions.write.mode(SaveMode.Overwrite).parquet(para.outputDir)
  }

}