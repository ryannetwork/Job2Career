package com.intel.analytics.bigdl.apps.job2Career

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.ml.feature.{LabeledPoint, StringIndexer}
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql._

import scala.collection.{immutable, mutable}
import scala.io.Source
import com.intel.analytics.bigdl.apps.job2Career.DataProcess._
import com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove.{doc2VecFromWordMap, loadWordVecMap, run}
import com.intel.analytics.bigdl.apps.recommendation.Utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.storage.StorageLevel
import scopt.OptionParser
import org.apache.spark.sql.functions._

import scala.util.Random

case class DataProcessParams(val inputDir: String = "/Users/guoqiong/intelWork/projects/jobs2Career/",
                             val outputDir: String = "add it if you need it",
                             val topK: Int = 500,
                             val negativeK: Int = 50,
                             val dictDir: String = "/Users/guoqiong/intelWork/projects/wrapup/textClassification/keras/glove.6B/glove.6B.300d.txt")

object DataProcess {

  def main(args: Array[String]): Unit = {

    val defaultParams = DataProcessParams()
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("app")

    val spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.sqlContext.setConf("spark.sql.shuffle.partitions", "1000")

    val parser = new OptionParser[DataProcessParams]("BigDL Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[String]("outputDir")
        .text(s"outputDir")
        .action((x, c) => c.copy(outputDir = x))
      opt[String]("topK")
        .text(s"topK")
        .action((x, c) => c.copy(topK = x.toInt))
      opt[String]("negativeK")
        .text(s"negativeK")
        .action((x, c) => c.copy(negativeK = x.toInt))
      opt[String]("dictDir")
        .text(s"wordVec data")
        .action((x, c) => c.copy(dictDir = x))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(spark, params)
    } getOrElse {
      System.exit(1)
    }

  }

  def run(spark: SparkSession, para: DataProcessParams): Unit = {

    val dataPath = para.inputDir
    val lookupDict = para.dictDir
    val dict: Map[String, Array[Float]] = loadWordVecMap(lookupDict)
    val br: Broadcast[Map[String, Array[Float]]] = spark.sparkContext.broadcast(dict)

    val applicationDF = spark.read.parquet(dataPath + "/resume_search/application_job_resume_2016_2017_10.parquet")
      .select("job_id", "jobs_description", "resume_url", "resume.resume.normalizedBody", "new_application")
      .withColumn("label", applicationStatusUdf(col("new_application")))
      .withColumn("userId", createResumeId(col("resume_url")))
      .withColumnRenamed("resume.resume.normalizedBody", "normalizedBody")
      .withColumnRenamed("normalizedBody", "userDoc")
      .withColumnRenamed("jobs_description", "itemDoc")
      .withColumnRenamed("job_id", "itemId")
      .select("userId", "itemId", "userDoc", "itemDoc", "label")

    //    println("---------read original data and select columns--------")
    //    println(applicationDF.count())
    //    applicationDF.printSchema()
    //    applicationDF.show(5)

    val applicationCleaned = cleanData(applicationDF, br.value)

    //    val userCount = applicationCleaned.select("userId").distinct().count()
    //    val itemCount = applicationCleaned.select("itemId").distinct().count()
    //    val appCount = applicationCleaned.count()
    //
    //    println("________________after cleaning______________________")
    //    println("userCount= " + userCount)
    //    println("itemCount= " + itemCount)
    //    println("appCount= " + appCount)

    val applicationVectors = getGloveVectors(applicationCleaned, br)

    val (indexed, userDict, itemDict) = indexData(applicationVectors)

    indexed.cache()
    userDict.cache()
    itemDict.cache()

    //    println("------------------------distribution in indexed -----------------------")
    //    indexed.groupBy("userIdIndex").count()
    //      .withColumnRenamed("count", "applyJobCount")
    //      .groupBy("applyJobCount").count()
    //      .orderBy("applyJobCount").show(1000, false)


    val output = dataPath + "/data/indexed_application_job_resume_2016_2017_10_300d"

    indexed.printSchema()

    applicationVectors.coalesce(16).write.mode(SaveMode.Overwrite).parquet(output + "/applicationVectors")
    indexed.coalesce(16).write.mode(SaveMode.Overwrite).parquet(output + "/indexed")
    userDict.write.mode(SaveMode.Overwrite).parquet(output + "/userDict")
    itemDict.write.mode(SaveMode.Overwrite).parquet(output + "/itemDict")
    // joined data write out

    //    val negativeDF = negativeJoin(indexed, itemDict, userDict, para.negativeK)
    //
    //    // println("after negative join " + negativeDF.count())
    //    negativeDF.coalesce(16).write.mode(SaveMode.Overwrite).parquet(output + "/NEG" + para.negativeK)
    //
    //    val joinAllDF = crossJoinAll(userDict, itemDict, indexed, para.topK)
    //    joinAllDF.write.mode(SaveMode.Overwrite).parquet(output + "/ALL")

    println("done")
  }

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

    val userVectors = applicationIndexed.select("userId", "userIdIndex", "userVec").distinct()
    val userDict = dedupe(userVectors, "userIdIndex", "userVec")

    val itemVectors = applicationIndexed.select("itemId", "itemIdIndex", "itemVec").distinct()
    val itemDict = dedupe(itemVectors, "itemIdIndex", "itemVec")

    (indexed, userDict, itemDict)
  }

  //for each id, average the glove vectors
  def dedupe(vecDict: DataFrame, indexCol: String, vecCol: String) = {

    val avg = udf((vecs: mutable.WrappedArray[mutable.WrappedArray[Float]]) => {

      val docVec: Array[Float] = vecs
        .flatMap(x => x.zipWithIndex.map(x => (x._2, x._1)))
        .groupBy(x => x._1)
        .map(x => (x._1, x._2.map(_._2).sum / vecs.length)).toArray
        .sortBy(x => x._1)
        .map(x => x._2)

      docVec

    })

    val colName = "collect_list(" + vecCol + ")"

    vecDict.groupBy(indexCol)
      .agg(collect_list(col(vecCol)))
      .withColumn(vecCol, avg(col(colName)))
      .drop(colName)

  }

  def negativeJoin(indexed: DataFrame, itemDict: DataFrame, userDict: DataFrame, negativeK: Int = 50): DataFrame = {

    val negativeSamples = getNegativeSamples2(negativeK, indexed)

    val unioned = negativeSamples.union(indexed)
    val out = unioned
      .join(userDict, unioned("userIdIndex") === userDict("userIdIndex"))
      .join(itemDict, unioned("itemIdIndex") === itemDict("itemIdIndex"))
      .select(userDict("userIdIndex"), itemDict("itemIdIndex"), col("label"), userDict("userVec"), itemDict("itemVec"))
      .withColumn("cosineSimilarity", getCosineSim(col("userVec"), col("itemVec")))
      .sort(col("cosineSimilarity").desc)
      .withColumnRenamed("cosineSimilarity", "score")

    out
  }

  def crossJoinAll(userDict: DataFrame, itemDict: DataFrame, indexed: DataFrame, K: Int = 50): DataFrame = {

    val outAll = userDict.select("userIdIndex", "userVec").
      crossJoin(itemDict.select("itemIdIndex", "itemVec"))
      .withColumn("score", getCosineSim(col("userVec"), col("itemVec")))
      .drop("itemVec", "userVec")

    outAll.persist(StorageLevel.DISK_ONLY)
    val w1 = Window.partitionBy("userIdIndex").orderBy(desc("score"))
    val rankDF = outAll.withColumn("rank", rank.over(w1)).where(col("rank") <= K)

    rankDF.join(indexed, Seq("userIdIndex", "itemIdIndex"), "leftouter")
  }

  //("userId", "itemId", "userDoc", "itemDoc", "label")
  def cleanData(applicationIn: DataFrame, dict: Map[String, Array[Float]]) = {

    val applicationDF = applicationIn
      .withColumn("itemDoc", removeHTMLTag(col("itemDoc")))
      .filter(col("itemDoc").isNotNull && col("itemId").isNotNull &&
        lengthUdf(col("itemDoc")) > 10 && filterDoc(dict)(col("itemDoc")))
      .filter(col("userDoc").isNotNull && col("userId").isNotNull
        && lengthUdf(col("userDoc")) > 10 && filterDoc(dict)(col("userDoc")))
      .filter(col("label").isNotNull)
      .select("userId", "itemId", "userDoc", "itemDoc", "label")
      .distinct()
    applicationDF
  }

  def getDataSetByCol(dataFrame: DataFrame, column: String): Set[String] = {
    dataFrame.select(column)
      .filter(col(column).isNotNull)
      .distinct()
      .rdd
      .map(row => row(0).toString).collect().toSet
  }

  //("userId", "itemId", "userDoc", "itemDoc","userVec","itemVec", "label")
  def getGloveVectors(df: DataFrame, br: Broadcast[Map[String, Array[Float]]]): DataFrame = {

    val userVecDF = doc2VecFromWordMap(df, br, "userVec", "userDoc")
      .filter(sizeFilter(col("userVec")))

    val itemVecDF = doc2VecFromWordMap(userVecDF, br, "itemVec", "itemDoc")
      .filter(sizeFilter(col("itemVec")))

    itemVecDF

  }

  def filterDoc(dict: Map[String, Array[Float]]) = {
    val func = (doc: String) => {
      val seq = doc.split("\n")
        .flatMap(x => x.split(" "))
        .filter(x => x.size > 1 && dict.contains(x))
      seq.length > 0
    }
    udf(func)
  }


}
