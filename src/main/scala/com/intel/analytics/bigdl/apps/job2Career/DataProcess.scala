package com.intel.analytics.bigdl.apps.job2Career

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql._

import scala.collection.{immutable, mutable}
import scala.io.Source
import com.intel.analytics.bigdl.apps.job2Career.DataProcess._
import com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove.{doc2VecFromWordMap, loadWordVecMap, run}
import com.intel.analytics.bigdl.apps.recommendation.Utils.{addNegativeSample, getCosineSim, sizeFilter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.storage.StorageLevel
import scopt.OptionParser

import scala.util.Random

case class DataProcessParams(val inputDir: String = "/Users/guoqiong/intelWork/projects/jobs2Career/",
                             val outputDir: String = "add it if you need it",
                             val topK:Int = 500,
                      val dictDir: String = "/Users/guoqiong/intelWork/projects/wrapup/textClassification/keras/glove.6B/glove.6B.50d.txt")


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

  def cleanData(applicationIn: DataFrame, dict: Set[String]) = {

    //TODO choose only one of the descriptions for both user and item

    val br = applicationIn.sqlContext.sparkContext.broadcast(dict)

    val filterDoc = udf((doc: String) => {
      val seq = doc.split("\n")
        .flatMap(x => x.split(" "))
        .filter(x => x.size > 1 && br.value.contains(x))

      seq.length > 0

    })

    val applicationDF = applicationIn.withColumnRenamed("job_id", "itemId")
      .withColumn("itemDoc", removeHTMLTag(col("description")))
      .filter(col("itemDoc").isNotNull && col("itemId").isNotNull &&
        lengthUdf(col("itemDoc")) > 10 && filterDoc(col("itemDoc")))
      .withColumn("userId", createResumeId(col("resume_url")))
      .withColumnRenamed("normalizedBody", "userDoc")
      .filter(col("userDoc").isNotNull && col("userId").isNotNull
        && lengthUdf(col("userDoc")) > 10 && filterDoc(col("userDoc")))
      .withColumn("label", applicationStatusUdf(col("new_application")))
      .filter(col("label").isNotNull)
      .select("userId", "itemId", "label", "userDoc", "itemDoc")
      .distinct()


    val userCount = applicationDF.select("userId").distinct().count()
    val itemCount = applicationDF.select("itemId").distinct().count()
    val appCount = applicationDF.count()

    println("________________after cleaning______________________")
    println("userCount= " + userCount)
    println("itemCount= " + itemCount)
    println("appCount= " + appCount)

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

  def indexData(applicationDF: DataFrame, br: Broadcast[Map[String, Array[Float]]]) = {

    val si1 = new StringIndexer().setInputCol("userId").setOutputCol("userIdIndex")
    val si2 = new StringIndexer().setInputCol("itemId").setOutputCol("itemIdIndex")

    val pipeline = new Pipeline().setStages(Array(si1, si2))
    val pipelineModel = pipeline.fit(applicationDF)
    val applicationIndexed: DataFrame = pipelineModel.transform(applicationDF)

    val indexed = applicationIndexed.select("userIdIndex", "itemIdIndex", "label").distinct()
    val userDict = applicationIndexed.select("userId", "userIdIndex", "userDoc").distinct()
    println("original count of userDict:" + userDict.select("userIdIndex").distinct().count())
    val userVecOrig = doc2VecFromWordMap(userDict, br, "userVec", "userDoc").filter(sizeFilter(col("userVec")))

    userVecOrig.filter(!sizeFilter(col("userVec"))).show(100, false)
    println(userVecOrig.filter(!sizeFilter(col("userVec"))).count())

    val userVecDict = dedupe(userVecOrig,"userIdIndex","userVec")
    println("filter count of userDict:" + userVecDict.select("userIdIndex").distinct().count())

    val itemDict = applicationIndexed.select("itemId", "itemIdIndex", "itemDoc").distinct()
    val itemVecDictOrig = doc2VecFromWordMap(itemDict, br, "itemVec", "itemDoc").filter(sizeFilter(col("itemVec")))
    val itemVecDict = dedupe(itemVecDictOrig,"itemIdIndex","itemVec")

    println("original count of itemDict:" + itemDict.select("itemIdIndex").distinct().count())
    println("original vec count of itemDict:" + itemVecDictOrig.select("itemIdIndex").distinct().count())

    itemVecDictOrig.filter(!sizeFilter(col("itemVec"))).show(100, false)
    println(itemVecDict.filter(!sizeFilter(col("itemVec"))).count())

    println("------------------------in indexed -----------------------")

    indexed.groupBy("userIdIndex").count()
      .withColumnRenamed("count", "applyJobCount")
      .groupBy("applyJobCount").count()
      .orderBy("applyJobCount").show(1000, false)

    (indexed, userVecDict, itemVecDict)
  }


  def dedupe(vecDict: DataFrame, indexCol:String, vecCol:String) = {

    val avg = udf((vecs:mutable.WrappedArray[mutable.WrappedArray[Float]])=> {


      val docVec: Array[Float] = vecs
        .flatMap(x => x.zipWithIndex.map(x => (x._2, x._1)))
        .groupBy(x => x._1)
        .map(x => (x._1, x._2.map(_._2).sum/vecs.length)).toArray
        .sortBy(x => x._1)
        .map(x => x._2)

      docVec

    })

    val colName = "collect_list(" + vecCol +")"

    vecDict.groupBy(indexCol)
           .agg(collect_list(col(vecCol)))
           .withColumn(vecCol,avg(col(colName)))
           .drop(colName)

  }

  def negativeJoin(indexed: DataFrame, itemDict: DataFrame, userDict: DataFrame): DataFrame = {

    val indexedWithNegative = addNegativeSample(50, indexed)

    println("------------------------in negative join after adding negative samples----------------------")
    indexedWithNegative.filter("label = 1").groupBy("userIdIndex").count()
      .withColumnRenamed("count", "applyJobCount")
      .groupBy("applyJobCount").count()
      .orderBy("applyJobCount").show(1000, false)

    println(" __________ total count before join with dicts__________")
    println(indexedWithNegative.count())
    println(" __________ total count before join with dicts label = 1__________")
    println(indexedWithNegative.filter("label = 1").count())

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

    println(" __________ total count after join with dicts__________")
    println(joined.count())
    println(" __________ total count after join with dicts label = 1__________")
    println(joined.filter("label = 1").count())

    joined
  }

  def crossJoinAll(userDict: DataFrame, itemDict: DataFrame,  indexed: DataFrame, K: Int = 50): DataFrame = {

    val outAll = userDict.select("userIdIndex", "userVec").
      crossJoin(itemDict.select("itemIdIndex", "itemVec"))
      .withColumn("score", getCosineSim(col("userVec"), col("itemVec")))
      .drop("itemVec", "userVec")

    outAll.persist(StorageLevel.DISK_ONLY)
    val w1 = Window.partitionBy("userIdIndex").orderBy(desc("score"))
    val rankDF = outAll.withColumn("rank", rank.over(w1)).where(col("rank") <= K).drop("rank")

    rankDF.join(indexed, Seq("userIdIndex", "itemIdIndex"), "leftouter")
  }

  def main(args: Array[String]): Unit = {

    val defaultParams = DataProcessParams()

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

  def run(para:DataProcessParams): Unit ={

    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[8]")
      .setAppName("app")
    val sc = SparkContext.getOrCreate(conf)

    val spark = SparkSession.builder().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val dataPath = para.inputDir
    val lookupDict = para.dictDir

    val applicationDF = spark.read.parquet(dataPath + "/resume_search/application_job_resume_2016_2017_10.parquet")
      .withColumnRenamed("jobs_description", "description")
      .select("job_id", "description", "resume_url", "resume.resume.normalizedBody", "new_application")
      .withColumnRenamed("resume.resume.normalizedBody", "normalizedBody")

    applicationDF.printSchema()

    val dict = loadWordVecMap(lookupDict)
    val br: Broadcast[Map[String, Array[Float]]] = spark.sparkContext.broadcast(dict)

    val dataPreprocess = new DataProcess

    val applicationCleaned = dataPreprocess
      .cleanData(applicationDF, dict.map(x => x._1).toSet)

    val (indexed, userDict, itemDict) = indexData(applicationCleaned,br)

    indexed.cache()
    userDict.cache()
    itemDict.cache()


    val output = dataPath + "/data/indexed_application_job_resume_2016_2017_10"

    indexed.printSchema()

    val Row(minUserIdIndex: Double, maxUserIdIndex: Double) = userDict.agg(min("userIdIndex"), max("userIdIndex")).head
    val Row(minItemIdIndex: Double, maxItemIdIndex: Double) = itemDict.agg(min("itemIdIndex"), max("itemIdIndex")).head

    //    println("indexed application count: " + indexed.count())
    //    println("indexed userIdIndex: " + indexed.select("userIdIndex").distinct().count())
    //    println("indexed itemIdIndex: " + indexed.select("itemIdIndex").distinct().count())
    //    println("userDict: " + userDict.distinct().count)
    //    println("userIdIndex min: " + minUserIdIndex + "max: " + maxUserIdIndex)
    //    println("itemDict " + itemDict.distinct().count)
    //    println("itemIdIndex min: " + minItemIdIndex + "max: " + maxItemIdIndex)

    indexed.coalesce(16).write.mode(SaveMode.Overwrite).parquet(output + "/indexed")
    userDict.write.mode(SaveMode.Overwrite).parquet(output + "/userDict")
    itemDict.write.mode(SaveMode.Overwrite).parquet(output + "/itemDict")

    // joined data write out

//    val negativeDF = negativeJoin(indexed, itemDict, userDict)
//
//    // println("after negative join " + negativeDF.count())
//    negativeDF.coalesce(16).write.mode(SaveMode.Overwrite).parquet(output + "/NEG50")

    val joinAllDF = crossJoinAll(userDict, itemDict, indexed, para.topK)
    joinAllDF.write.mode(SaveMode.Overwrite).parquet(output + "/ALL")

    println("done")
  }

}
