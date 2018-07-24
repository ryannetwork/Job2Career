//package com.intel.analytics.bigdl.apps.job2Career
//
//import com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove.loadWordVecMap
//import com.intel.analytics.bigdl.apps.job2Career.TrainWithEnsambleNCF_Glove.kmeansProcessDF
//import com.intel.analytics.bigdl.apps.recommendation.{Evaluation, ModelParam, ModelUtils}
//import com.intel.analytics.bigdl.apps.recommendation.Utils.{add1, getFeaturesLP, toDecimal, toZero}
//import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
//import com.intel.analytics.bigdl.optim.Adam
//import com.intel.analytics.zoo.pipeline.nnframes.{NNClassifier, NNClassifierModel, NNEstimator, NNModel}
//import org.apache.hadoop.conf.Configuration
//import org.apache.hadoop.fs.Path
//import org.apache.spark.broadcast.Broadcast
//import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
//import org.apache.spark.mllib.linalg
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.functions.col
//import org.apache.spark.sql.{DataFrame, Row, SQLContext, SaveMode}
//import org.apache.spark.sql.types.IntegerType
//
//object TMP {
//
//  def kmeansProcessRDD(dataFrame: DataFrame,
//                       vectorCol: String,
//                       kmeanPath: String,
//                       isTrain: Boolean = true,
//                       numClusters: Int = 2,
//                       numIterations: Int = 20): DataFrame = {
//
//    val dataRdd: RDD[Row] = dataFrame.rdd
//
//    val trainData: RDD[linalg.Vector] = dataRdd.map(row => {
//      val d = row.getAs[scala.collection.mutable.WrappedArray[Float]](vectorCol).map(x => x.toDouble)
//      Vectors.dense(d.toArray)
//    })
//
//    val sc = dataFrame.sqlContext.sparkContext
//
//    val model: KMeansModel = if (isTrain) {
//      require(kmeanPath != null, "model path can't be null")
//      val cluster = KMeans.train(trainData, numClusters, numIterations)
//
//      val p = new Path(kmeanPath)
//      val fs = p.getFileSystem(new Configuration())
//      if (fs.exists(p)) {
//        fs.delete(p, true)
//      }
//
//      cluster.save(sc, kmeanPath)
//      cluster
//    } else {
//      KMeansModel.load(sc, kmeanPath)
//    }
//
//    // Evaluate clustering by computing Within Set Sum of Squared Errors
//
//    val WSSSE = model.computeCost(trainData)
//    println("WSSE:" + WSSSE)
//
//    val clusterRdd = model.predict(trainData)
//
//    val finalSchema = dataFrame.schema.add("kth", IntegerType)
//
//    val predictions = dataFrame.sqlContext.createDataFrame(dataRdd.zip(clusterRdd)
//      .map(x => Row.fromSeq(x._1.toSeq ++ Array[Any](x._2))), finalSchema)
//
//    predictions
//  }
//
//  def trainNCF(indexedWithNegative: DataFrame, param: TrainParam, modelPath: String, isTrain: Boolean = true) = {
//
//    val Array(trainWithNegative, validationWithNegative) = indexedWithNegative.randomSplit(Array(0.8, 0.2), 1L)
//
//    println("---------distribution of label in NCF for training and testing ----------------")
//    val trainingDF: DataFrame = getFeaturesLP(trainWithNegative)
//    val validationDF = getFeaturesLP(validationWithNegative)
//
//    trainingDF.persist()
//    trainingDF.groupBy("label").count().show()
//    validationDF.persist()
//    validationDF.groupBy("label").count().show()
//
//    val time1 = System.nanoTime()
//    val modelParam = ModelParam(userEmbed = 50,
//      itemEmbed = 50,
//      hiddenLayers = Array(40, 20, 10),
//      labels = 2)
//
//    val model = if (isTrain) {
//      val recModel = new ModelUtils(modelParam)
//
//      // val model = recModel.ncf(userCount, itemCount)
//      val model = recModel.mlp3
//
//      val criterion = ClassNLLCriterion[Float]()
//
//      val dlc = NNClassifier(model, criterion, Array(2 * param.vectDim))
//        .setBatchSize(param.batchSize)
//        .setOptimMethod(new Adam())
//        .setLearningRate(param.learningRate)
//        .setLearningRateDecay(param.learningRateDecay)
//        .setMaxEpoch(param.nEpochs)
//
//      val dlModel: NNModel[Float] = dlc.fit(trainingDF)
//
//      println(dlModel.model.getParameters())
//      dlModel.model.saveModule(modelPath, null, true)
//
//      dlModel
//    } else {
//
//      val loadedModel = Module.loadModule(modelPath, null)
//      println(loadedModel.getParameters())
//      val dlModel = NNClassifierModel(loadedModel, Array(2 * param.vectDim))
//      //  .setBatchSize(param.batchSize)
//
//      dlModel
//    }
//
//    val time2 = System.nanoTime()
//
//    val predictions: DataFrame = model.transform(trainingDF).cache()
//
//    predictions.show()
//    val time3 = System.nanoTime()
//
//    println(predictions.count())
//    println("validation results:" + modelPath)
//    Evaluation.evaluate2(predictions.withColumn("label", toZero(col("label")))
//      .withColumn("prediction", toZero(col("prediction"))))
//
//    val time4 = System.nanoTime()
//
//    val trainingTime = (time2 - time1) * (1e-9)
//    val predictionTime = (time3 - time2) * (1e-9)
//    val evaluationTime = (time4 - time3) * (1e-9)
//
//    println("training time(s):  " + toDecimal(3)(trainingTime))
//    println("prediction time(s):  " + toDecimal(3)(predictionTime))
//    println("evaluation time(s):  " + toDecimal(3)(evaluationTime))
//
//    trainingDF.unpersist()
//    validationDF.unpersist()
//  }
//
//
//  def processGoldendata(sqlContext: SQLContext, para: TrainParam, modelPath: String, numClusters: Int = 2) = {
//
//    val validationIn = sqlContext.read.parquet(para.valDir)
//    // validationIn.printSchema()
//    val validationDF = validationIn
//      .select("resume_id", "job_id", "resume.resume.normalizedBody", "description", "apply_flag")
//      .withColumn("label", add1(col("apply_flag")))
//      .withColumnRenamed("resume.resume.normalizedBody", "normalizedBody")
//      .withColumnRenamed("resume_id", "userId")
//      .withColumnRenamed("normalizedBody", "userDoc")
//      .withColumnRenamed("description", "itemDoc")
//      .withColumnRenamed("job_id", "itemId")
//      .select("userId", "itemId", "userDoc", "itemDoc", "label")
//      .filter(col("itemDoc").isNotNull && col("userDoc").isNotNull && col("userId").isNotNull
//        && col("itemId").isNotNull && col("label").isNotNull)
//
//    val dict: Map[String, Array[Float]] = loadWordVecMap(para.dictDir)
//    val br: Broadcast[Map[String, Array[Float]]] = sqlContext.sparkContext.broadcast(dict)
//
//    val validationCleaned = DataProcess.cleanData(validationDF, br.value)
//    val validationVectors: DataFrame = DataProcess.getGloveVectors(validationCleaned, br)
//
//    val validationVectorsCluster = kmeansProcessDF(validationVectors, "userVec", para.inputDir + "/modelKmean", false, numClusters).persist()
//
//    val predictorDFS = (0 to numClusters - 1).map(kth => {
//
//      val loadedModel = Module.loadModule(modelPath + kth, null)
//      val dlModel = NNClassifierModel[Float](loadedModel, Array(2 * para.vectDim))
//        .setBatchSize(para.batchSize)
//
//      val validationVectorsKth = validationVectorsCluster.filter("kth = " + kth)
//
//      //step2
//      // every point go through the K mean model, and get a K for each record
//      //KmeanModel.predict(validationVectors), userVector renames features, transform kth and evaluate each model, return metrics
//
//      val validationLP = getFeaturesLP(validationVectorsKth).coalesce(32)
//
//      val predictions2: DataFrame = dlModel.transform(validationLP)
//
//      predictions2.select("userId", "itemId", "label", "prediction").show(20, false)
//
//      predictions2
//
//    })
//
//    validationVectorsCluster.unpersist()
//
//    val predictorDF = predictorDFS.reduce((x, y) => x.union(y))
//    predictorDF.write.mode(SaveMode.Overwrite).parquet(para.outputDir)
//    // summarize
//    println("validation results on golden dataset")
//    val dataToValidation = predictorDF.withColumn("label", toZero(col("label")))
//      .withColumn("prediction", toZero(col("prediction")))
//    Evaluation.evaluate2(dataToValidation)
//  }
//
//}
