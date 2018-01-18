package com.intel.analytics.bigdl.apps.job2Career

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql._

import scala.collection.immutable
import scala.io.Source
import com.intel.analytics.bigdl.apps.job2Career.DataProcess._
import com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove.{doc2VecFromWordMap, loadWordVecMap}
import com.intel.analytics.bigdl.apps.recommendation.Utils.{addNegativeSample, getCosineSim, sizeFilter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.expressions.Window

import scala.util.Random

class DataProcess {

  def cleanAndJoinData(applicationIn: DataFrame, userIn: DataFrame, itemIn: DataFrame) = {

    //TODO choose one of the descriptions for both user and item
    val userDF = userIn.select("resume_url", "resume.resume.normalizedBody")
      .withColumnRenamed("normalizedBody", "userDoc")
      .withColumn("userId", createResumeId(col("resume_url")))
      .filter(col("userDoc").isNotNull && col("userId").isNotNull && lengthUdf(col("userDoc")) > 10)
      .select("userId", "userDoc")

    val itemDF = itemIn.withColumnRenamed("job_id", "itemId")
      .withColumn("itemDoc", removeHTMLTag(col("description")))
      .filter(col("itemDoc").isNotNull && col("itemId").isNotNull && lengthUdf(col("itemDoc")) > 10)
      .select("itemId", "itemDoc")

    val applicationDF = applicationIn.withColumnRenamed("job_id", "itemId")
      .withColumn("userId", createResumeId(col("resume_url")))
      .withColumn("label", applicationStatusUdf(col("new_application")))
      .filter(col("itemId").isNotNull && col("userId").isNotNull)
      .select("userId", "itemId", "label")
      .distinct()

    println("application count" + applicationDF.count())


    val applicationOut = applicationDF.join(userDF, applicationDF("userId") === userDF("userId"))
      .join(itemDF, applicationDF("itemId") === itemDF("itemId"))
      .drop(userDF("userId"))
      .drop(itemDF("itemId"))
      .distinct()

    //TODO to improve performance of joining tables when data is too big

    //    val userIdSet = getDataSetByCol(userDF,"userId")
    //    val userIdSetUdf = udf((userId: String) => userIdSet.contains(userId))
    //
    //    val itemIdSet = getDataSetByCol(itemDF,"itemId")
    //    val itemIdSetUdf = udf((job_Id: Long) => itemIdSet.contains(job_Id.toString))
    //
    //    val applicationOut = applicationDF
    //      .filter(col("userId").isNotNull)
    //      .filter(itemIdSetUdf(col("itemId")) && userIdSetUdf(col("userId"))).distinct()
    println("application count after join" + applicationOut.count())
    applicationOut
  }

  def cleanData(applicationIn: DataFrame) = {

    //TODO choose only one of the descriptions for both user and item

    val applicationDF = applicationIn.withColumnRenamed("job_id", "itemId")
      .withColumn("itemDoc", removeHTMLTag(col("description")))
      .filter(col("itemDoc").isNotNull && col("itemId").isNotNull && lengthUdf(col("itemDoc")) > 10)
      .withColumn("userId", createResumeId(col("resume_url")))
      .withColumnRenamed("normalizedBody", "userDoc")
      .filter(col("userDoc").isNotNull && col("userId").isNotNull && lengthUdf(col("userDoc")) > 10)
      .withColumn("label", applicationStatusUdf(col("new_application")))
      .filter(col("label").isNotNull)
      .select("userId", "itemId", "label", "userDoc", "itemDoc")
      .distinct()

    println("application count" + applicationDF.count())

    applicationDF
  }

  def getDataSetByCol(dataFrame: DataFrame, column: String): Set[String] = {
    dataFrame.select(column)
      .filter(col(column).isNotNull)
      .distinct()
      .rdd
      .map(row => row(0).toString).collect().toSet
  }

}

object DataProcess {

  def apply: DataProcess = new DataProcess()

  val lengthUdf = udf((doc: String) => doc.length)

  val removeHTMLTag = udf((str: String) => str.replaceAll("\\<.*?>", ""))

  val createResumeId = udf((url: String) => {
    if (url == null || url.trim == "") {
      null
    } else {
      val lastSlash = url.lastIndexOf("/")
      val result: String = url.substring(Math.min(Math.max(lastSlash, 0) + 3, url.length - 1))
      result.replace("pdf", "").replace("docx", "").replace("doc", "").replace("txt", "")
    }
  })

  val applicationStatusUdf = udf((application: Boolean) => 1.0)

  def indexData(applicationDF: DataFrame) = {

    val si1 = new StringIndexer().setInputCol("userId").setOutputCol("userIdIndex")
    val si2 = new StringIndexer().setInputCol("itemId").setOutputCol("itemIdIndex")

    val pipeline = new Pipeline().setStages(Array(si1, si2))
    val pipelineModel = pipeline.fit(applicationDF)
    val applicationIndexed: DataFrame = pipelineModel.transform(applicationDF)

    val indexed = applicationIndexed.select("userIdIndex", "itemIdIndex", "label").distinct()
    val userDict = applicationIndexed.select("userId", "userIdIndex", "userDoc").distinct()
    val itemDict = applicationIndexed.select("itemId", "itemIdIndex", "itemDoc").distinct()

    println("------------------------in indexed ----------------------")

    indexed.groupBy("userIdIndex").count()
      .withColumnRenamed("count", "applyJobCount")
      .groupBy("applyJobCount").count()
      .orderBy("applyJobCount").show(1000, false)

    (indexed, userDict, itemDict)
  }

  def negativeJoin(indexed: DataFrame, itemDictOrig: DataFrame, userDictOrig: DataFrame, br: Broadcast[Map[String, Array[Float]]]): DataFrame = {

    println("original count of userDict:" + userDictOrig.select("userIdIndex").distinct().count())
    val userDict = doc2VecFromWordMap(userDictOrig, br, "userVec", "userDoc")
      .filter(sizeFilter(col("userVec")))

    userDict.filter(!sizeFilter(col("userVec"))).show(100, false)

    println("filter count of userDict:" + userDict.select("userIdIndex").distinct().count())


    println("original count of itemDict:" + itemDictOrig.select("itemIdIndex").distinct().count())
    val itemDict = doc2VecFromWordMap(itemDictOrig, br, "itemVec", "itemDoc")
      .filter(sizeFilter(col("itemVec")))
    println("original count of itemDict:" + itemDict.select("itemIdIndex").distinct().count())

    val indexedWithNegative = addNegativeSample(50, indexed)

    println("------------------------in negative join after adding negative samples----------------------")
    indexedWithNegative.filter("label = 1").groupBy("userIdIndex").count()
      .withColumnRenamed("count", "applyJobCount")
      .groupBy("applyJobCount").count()
      .orderBy("applyJobCount").show(1000, false)


    val joined = indexedWithNegative
      .join(userDict, indexedWithNegative("userIdIndex") === userDict("userIdIndex"))
      .join(itemDict, indexedWithNegative("itemIdIndex") === itemDict("itemIdIndex"))
      .select(userDict("userIdIndex"), itemDict("itemIdIndex"), col("label"), col("userVec"), col("itemVec"))
      .withColumn("cosineSimilarity", getCosineSim(col("userVec"), col("itemVec")))
      .sort(col("cosineSimilarity").desc)
      .withColumnRenamed("cosineSimilarity", "score")
      .select("userIdIndex", "itemIdIndex", "score", "label")

    println("------------------------in negative join afterscore----------------------")

    joined.filter("label = 1").groupBy("userIdIndex").count()
      .withColumnRenamed("count", "applyJobCount")
      .groupBy("applyJobCount").count()
      .orderBy("applyJobCount").show(1000, false)

    joined

  }


  def crossJoinAll(userDictOrig: DataFrame, itemDictOrig: DataFrame, br: Broadcast[Map[String, Array[Float]]], indexed: DataFrame, K: Int = 50): DataFrame = {

    val userDict = doc2VecFromWordMap(userDictOrig, br, "userVec", "userDoc")
      .filter(sizeFilter(col("userVec")))

    val itemDict = doc2VecFromWordMap(itemDictOrig, br, "itemVec", "itemDoc")
      .filter(sizeFilter(col("itemVec")))

    val outAll = userDict.select("userIdIndex", "userVec").
      crossJoin(itemDict.select("itemIdIndex", "itemVec"))
      .withColumn("score", getCosineSim(col("userVec"), col("itemVec")))
      .drop("itemVec", "userVec")

    val w1 = Window.partitionBy("userIdIndex").orderBy(desc("score"))
    val rankDF = outAll.withColumn("rank", rank.over(w1)).where(col("rank") <= K).drop("rank")

    rankDF.join(indexed, Seq("userIdIndex", "itemIdIndex"), "leftouter")
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[8]")
      .setAppName("app")
    val sc = SparkContext.getOrCreate(conf)

    val spark = SparkSession.builder().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val dataPath = "/Users/guoqiong/intelWork/projects/jobs2Career/"
    val lookupDict = "/Users/guoqiong/intelWork/projects/wrapup/textClassification/keras/glove.6B/glove.6B.50d.txt"

    val applicationDF = spark.read.parquet(dataPath + "/resume_search/application_job_resume_2016_2017_10.parquet")
      .withColumnRenamed("jobs_description", "description")
      .select("job_id", "description", "resume_url", "resume.resume.normalizedBody", "new_application")
      .withColumnRenamed("resume.resume.normalizedBody", "normalizedBody")

    applicationDF.printSchema()

    val dataPreprocess = new DataProcess

    val applicationCleaned = dataPreprocess
      .cleanData(applicationDF)

    val (indexed, userDict, itemDict) = indexData(applicationCleaned)

    indexed.cache()
    userDict.cache()
    itemDict.cache()


    val output = dataPath + "/data/indexed_application_job_resume_2016_2017_10"

    indexed.printSchema()
    println("indexed application count: " + indexed.count())
    println("indexed userIdIndex: " + indexed.select("userIdIndex").distinct().count())
    println("indexed itemIdIndex: " + indexed.select("itemIdIndex").distinct().count())

    val Row(minUserIdIndex: Double, maxUserIdIndex: Double) = userDict.agg(min("userIdIndex"), max("userIdIndex")).head
    val Row(minItemIdIndex: Double, maxItemIdIndex: Double) = itemDict.agg(min("itemIdIndex"), max("itemIdIndex")).head

    println("userIdIndex: " + userDict.count)
    println("userIdIndex min: " + minUserIdIndex + "max: " + maxUserIdIndex)
    println("itemIdIndex min: " + minItemIdIndex + "max: " + maxItemIdIndex)

    println("itemDict " + itemDict.count)
    println("itemDict " + itemDict.distinct().count)

    indexed.coalesce(16).write.mode(SaveMode.Overwrite).parquet(output + "/indexed")
    userDict.write.mode(SaveMode.Overwrite).parquet(output + "/userDict")
    itemDict.write.mode(SaveMode.Overwrite).parquet(output + "/itemDict")

    // joined data write out

    val dict = loadWordVecMap(lookupDict)
    val br: Broadcast[Map[String, Array[Float]]] = spark.sparkContext.broadcast(dict)


    val negativeDF = negativeJoin(indexed, itemDict, userDict, br)
    negativeDF.coalesce(16).write.mode(SaveMode.Overwrite).parquet(output + "/NEG50")

    //    val joinAllDF = crossJoinAll(userDict, itemDict, br, indexed, 100)
    //    joinAllDF.write.mode(SaveMode.Overwrite).parquet(output + "/ALL")

  }

}
