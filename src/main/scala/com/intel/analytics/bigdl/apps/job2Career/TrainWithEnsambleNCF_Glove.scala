package com.intel.analytics.bigdl.apps.job2Career

import com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove.loadWordVecMap
import com.intel.analytics.bigdl.apps.recommendation.Utils._
import com.intel.analytics.bigdl.apps.recommendation.{Evaluation, ModelParam, ModelUtils}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.nnframes.{NNClassifier, NNClassifierModel, NNEstimator, NNModel}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import scopt.OptionParser
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.clustering.{KMeans => MLKmeans, KMeansModel => MLKmeansModel}
import org.apache.spark.ml.linalg.{Vectors => MLVectors}
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object TrainWithEnsambleNCF_Glove {

  def main(args: Array[String]): Unit = {

    val defaultParams = TrainParam(
      inputDir = "/Users/guoqiong/intelWork/projects/jobs2Career/data/indexed_application_job_resume_2016_2017_10",
      outputDir = "/Users/guoqiong/intelWork/projects/jobs2Career/data/validation_predict",
      dictDir = "/Users/guoqiong/intelWork/projects/wrapup/textClassification/keras/glove.6B/glove.6B.50d.txt",
      valDir = "/Users/guoqiong/intelWork/projects/jobs2Career/data/validation/part*",
      batchSize = 8000,
      nEpochs = 10,
      vectDim = 50,
      learningRate = 0.005,
      learningRateDecay = 1e-6,
      Kclusters = 3)

    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("jobs2career").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val parser = new OptionParser[TrainParam]("jobs2career") {
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
      opt[Int]('f', "vectDim")
        .text("dimension of glove vectors")
        .action((x, c) => c.copy(vectDim = x))
      opt[Double]('l', "learningRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x.toDouble))
      opt[Double]('d', "learningRateDecay")
        .text("learning rate decay")
        .action((x, c) => c.copy(learningRateDecay = x.toDouble))
      opt[Double]('K', "Kclusters")
        .text("Kclusters")
        .action((x, c) => c.copy(Kclusters = x.toInt))

    }

    parser.parse(args, defaultParams).map {
      params =>
        run(sqlContext: SQLContext, params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(sqlContext: SQLContext, param: TrainParam): Unit = {

    val input: String = param.inputDir
    val ncfModelPath = param.inputDir + "/ncf"

    val userDict = sqlContext.read.parquet(input + "/userDict")
    val itemDict = sqlContext.read.parquet(input + "/itemDict")

    val numClusters = param.Kclusters
    // val kmeanUserDict = kmeansProcessRDD(userDict, "userVec", param.inputDir + "/modelKmean", true, numClusters).persist()
    //kmeanUserDict.write.mode(SaveMode.Overwrite).parquet(param.inputDir + "/userDictWithKRDD")

    val kmeanUserDict = sqlContext.read.parquet(param.inputDir + "/userDictWithKRDD")

    println("kmean kth cluster num distribution")
    kmeanUserDict.select("kth").groupBy("kth").count().show()
    val indexedWithNegative = sqlContext.read.parquet(input + "/indexedNeg1")

    (0 to numClusters - 1).map(kth => {

      val userDictKth = kmeanUserDict.filter(s"kth = $kth")

      val indexedKth = indexedWithNegative.join(userDictKth.select("userIdIndex"), Seq("userIdIndex")).cache()

      println("--------------------------------------------------")
      println(kth + " cluster data distribution: ")
      indexedKth.groupBy("label").count().show()

      trainNCF(indexedKth, param, ncfModelPath + "/" + kth)
    })

    kmeanUserDict.unpersist()

    // processGoldendata(spark, param, modelPath, numClusters)
  }


  def kmeansProcessDF(dataset: DataFrame,
                      vectorCol: String,
                      kmeanPath: String,
                      isTrain: Boolean = true,
                      numClusters: Int = 2,
                      numIterations: Int = 20): DataFrame = {

    val array2vec = udf((arr: scala.collection.mutable.WrappedArray[Float]) => {
      val d = arr.map(x => x.toDouble)
      MLVectors.dense(d.toArray)
    })

    val dataml = dataset.withColumn("features", array2vec(col("userVec")))
      .drop("userVec")
      .withColumnRenamed("features", "userVec")
    val kmeans = new MLKmeans()
      .setK(numClusters)
      .setSeed(1L)
      .setInitMode("k-means||")
      .setMaxIter(numIterations)
      .setFeaturesCol("userVec")
      .setPredictionCol("kth")

    // val clusters: MLKmeansModel = kmeans.fit(dataset)

    val model: MLKmeansModel = if (isTrain) {
      require(kmeanPath != null, "model path can't be null")
      val trained: MLKmeansModel = kmeans.fit(dataml)

      val p = new Path(kmeanPath)
      val fs = p.getFileSystem(new Configuration())
      if (fs.exists(p)) {
        fs.delete(p, true)
      }

      trained.save(kmeanPath)
      trained
    } else {
      MLKmeansModel.load(kmeanPath)
    }

    val predictions = model.transform(dataml)
    val evaluator = new ClusteringEvaluator()
      .setFeaturesCol("userVec").setPredictionCol("kth")

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

    predictions
  }

  def kmeansProcessRDD(dataFrame: DataFrame,
                       vectorCol: String,
                       kmeanPath: String,
                       isTrain: Boolean = true,
                       numClusters: Int = 2,
                       numIterations: Int = 20): DataFrame = {

    val dataRdd: RDD[Row] = dataFrame.rdd

    val trainData: RDD[linalg.Vector] = dataRdd.map(row => {
      val d = row.getAs[scala.collection.mutable.WrappedArray[Float]](vectorCol).map(x => x.toDouble)
      Vectors.dense(d.toArray)
    })

    val sc = dataFrame.sqlContext.sparkContext

    val model: KMeansModel = if (isTrain) {
      require(kmeanPath != null, "model path can't be null")
      val cluster = KMeans.train(trainData, numClusters, numIterations)

      val p = new Path(kmeanPath)
      val fs = p.getFileSystem(new Configuration())
      if (fs.exists(p)) {
        fs.delete(p, true)
      }

      cluster.save(sc, kmeanPath)
      cluster
    } else {
      KMeansModel.load(sc, kmeanPath)
    }

    // Evaluate clustering by computing Within Set Sum of Squared Errors

    val WSSSE = model.computeCost(trainData)
    println("WSSE:" + WSSSE)

    val clusterRdd = model.predict(trainData)

    val finalSchema = dataFrame.schema.add("kth", IntegerType)

    val predictions = dataFrame.sqlContext.createDataFrame(dataRdd.zip(clusterRdd)
      .map(x => Row.fromSeq(x._1.toSeq ++ Array[Any](x._2))), finalSchema)

    predictions
  }

  def trainNCF(indexedWithNegative: DataFrame, param: TrainParam, modelPath: String, isTrain: Boolean = true) = {

    val Array(trainWithNegative, validationWithNegative) = indexedWithNegative.randomSplit(Array(0.8, 0.2), 1L)

    println("---------distribution of label in NCF for training and testing ----------------")
    val trainingDF: DataFrame = getFeaturesLP(trainWithNegative)
    val validationDF = getFeaturesLP(validationWithNegative)

    trainingDF.groupBy("label").count().show()
    trainingDF.persist()
    validationDF.groupBy("label").count().show()
    validationDF.persist()

    val time1 = System.nanoTime()
    val modelParam = ModelParam(userEmbed = 20,
      itemEmbed = 20,
      midLayers = Array(40, 20),
      labels = 2)

    val model = if (isTrain) {
      val recModel = new ModelUtils(modelParam)

      // val model = recModel.ncf(userCount, itemCount)
      val model = recModel.mlp3

      val criterion = ClassNLLCriterion()

      val dlc: NNEstimator[Float] = NNClassifier[Float](model, criterion, Array(2 * param.vectDim))
        .setBatchSize(param.batchSize)
        .setOptimMethod(new Adam())
        .setLearningRate(param.learningRate)
        .setLearningRateDecay(param.learningRateDecay)
        .setMaxEpoch(param.nEpochs)

      val dlModel: NNModel[Float] = dlc.fit(trainingDF)

      println(dlModel.model.getParameters())
      dlModel.model.saveModule(modelPath, null, true)

      dlModel
    } else {

      val loadedModel = Module.loadModule(modelPath, null)
      println(loadedModel.getParameters())
      val dlModel = NNClassifierModel[Float](loadedModel, Array(2 * param.vectDim))
        .setBatchSize(param.batchSize)

      dlModel
    }

    val time2 = System.nanoTime()

    val predictions: DataFrame = model.transform(trainingDF).cache()

    predictions.show()
    val time3 = System.nanoTime()

    println(predictions.count())
    println("validation results:" + modelPath)
    Evaluation.evaluate2(predictions.withColumn("label", toZero(col("label")))
      .withColumn("prediction", toZero(col("prediction"))))

    val time4 = System.nanoTime()

    val trainingTime = (time2 - time1) * (1e-9)
    val predictionTime = (time3 - time2) * (1e-9)
    val evaluationTime = (time4 - time3) * (1e-9)

    println("training time(s):  " + toDecimal(3)(trainingTime))
    println("prediction time(s):  " + toDecimal(3)(predictionTime))
    println("evaluation time(s):  " + toDecimal(3)(predictionTime))
  }


  def processGoldendata(sqlContext: SQLContext, para: TrainParam, modelPath: String, numClusters: Int = 2) = {

    val validationIn = sqlContext.read.parquet(para.valDir)
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
    val br: Broadcast[Map[String, Array[Float]]] = sqlContext.sparkContext.broadcast(dict)

    val validationCleaned = DataProcess.cleanData(validationDF, br.value)
    val validationVectors: DataFrame = DataProcess.getGloveVectors(validationCleaned, br)

    val validationVectorsCluster = kmeansProcessRDD(validationVectors, "userVec", para.inputDir + "/modelKmean", false, numClusters).persist()

    val predictorDFS = (0 to numClusters - 1).map(kth => {

      val loadedModel = Module.loadModule(modelPath + kth, null)
      val dlModel = NNClassifierModel[Float](loadedModel, Array(2 * para.vectDim))
        .setBatchSize(para.batchSize)

      val validationVectorsKth = validationVectorsCluster.filter("kth = " + kth)

      //step2
      // every point go through the K mean model, and get a K for each record
      //KmeanModel.predict(validationVectors), userVector renames features, transform kth and evaluate each model, return metrics

      val validationLP = getFeaturesLP(validationVectorsKth).coalesce(32)

      val predictions2: DataFrame = dlModel.transform(validationLP)

      predictions2.select("userId", "itemId", "label", "prediction").show(20, false)

      predictions2

    })

    validationVectorsCluster.unpersist()

    val predictorDF = predictorDFS.reduce((x, y) => x.union(y))
    predictorDF.write.mode(SaveMode.Overwrite).parquet(para.outputDir)
    // summarize
    println("validation results on golden dataset")
    val dataToValidation = predictorDF.withColumn("label", toZero(col("label")))
      .withColumn("prediction", toZero(col("prediction")))
    Evaluation.evaluate2(dataToValidation)
  }

}