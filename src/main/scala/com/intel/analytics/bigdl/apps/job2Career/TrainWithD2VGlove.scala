package com.intel.analytics.bigdl.apps.job2Career

import com.intel.analytics.bigdl.apps.job2Career.DataProcess.{createResumeId, removeHTMLTag}
import com.intel.analytics.bigdl.apps.recommendation.Utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

import scala.collection.immutable
import scala.io.Source


object TrainWithD2VGlove {

  val gloveDir = s"/glove.6B/"
  val resumePath = "/"
  val jobPath = "/"

  val stopWordSet = Set("is", "the", "are", "we")

  def loadWordVecMap(filename: String): Map[String, Array[Float]] = {
    val wordMap = for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) yield {
      val values = line.split(" ")
      val word = values(0)
      val coefs: Array[Float] = values.slice(1, values.length).map(_.toFloat)
      (word, coefs)
    }
    wordMap.toMap
  }

  def doc2VecFromWordMap(dataFrame: DataFrame, brMap: Broadcast[Map[String, Array[Float]]], newCol: String, colName: String) = {

    val createVectorUdf = udf((doc: String) => {
      /**
        * x y z x
        **/
      val seq = doc.split(" ").filter(x => x.size > 1)

      val wordCount = seq
        .map(x => (x, 1))
        .groupBy(x => x._1)
        .map(x => (x._1, x._2.map(y => y._2).sum))

      val n = seq.length

      val wordFreqVec: immutable.Iterable[Array[Float]] = wordCount
        .filter(p => brMap.value.contains(p._1))
        .map(x => brMap.value(x._1).map(v => v * x._2 / n)
        )

      val docVec: Array[Float] = wordFreqVec
        .flatMap(x => x.zipWithIndex.map(x => (x._2, x._1)))
        .groupBy(x => x._1)
        .map(x => (x._1, x._2.map(_._2).sum)).toArray
        .sortBy(x => x._1)
        .map(x => x._2)

      docVec

    })

    dataFrame.withColumn(newCol, createVectorUdf(col(colName)))
  }


  def createDocVec(sqlContext: SQLContext): (DataFrame, DataFrame) = {

    val lookupMap = loadWordVecMap(gloveDir)

    val brMap = sqlContext.sparkContext.broadcast(lookupMap)

    val jobDF = sqlContext.read.parquet(jobPath)
      .withColumn("description", removeHTMLTag(col("description")))
    val jobVecDF = doc2VecFromWordMap(jobDF, brMap, "job_vec", "description")
      .select("jobid", "job_vec")

    val resumeDF = sqlContext.read.parquet(resumePath)
      .withColumn("resume_id", createResumeId(col("resume_url")))
      .select("resume_id", "resume.resume.body")
      .withColumnRenamed("resume.resume.body", "body")

    val resumeVecDF = doc2VecFromWordMap(resumeDF, brMap, "resume_vec", "body").select("resume_id", "resume_vec")

    (jobVecDF, resumeVecDF)

  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    // val input = "/Users/guoqiong/intelWork/projects/jobs2Career/data/"
    val input = "/Users/guoqiong/intelWork/projects/jobs2Career/data/indexed_application_job_resume_2016_2017_10"

    val lookupDict = "/Users/guoqiong/intelWork/projects/wrapup/textClassification/keras/glove.6B/glove.6B.50d.txt"

    val dict = loadWordVecMap(lookupDict)
    val br = spark.sparkContext.broadcast(dict)

    val indexed = spark.read.parquet(input + "/indexed")
    val indexedWithNegative = addNegativeSample(indexed)

    val userDict_temp = spark.read.parquet(input + "/userDict")
      .filter(col("userDoc").isNotNull)
    val userDict = doc2VecFromWordMap(userDict_temp, br, "userVec", "userDoc")
      .filter(sizeFilter(col("userVec")))

    val itemDict_temp = spark.read.parquet(input + "/itemDict")
      .filter(col("itemDoc").isNotNull)

    val itemDict = doc2VecFromWordMap(itemDict_temp, br, "itemVec", "itemDoc")
      .filter(sizeFilter(col("itemVec")))

    val time1 = System.nanoTime()

    val joined = indexedWithNegative
      .join(userDict, indexedWithNegative("userIdIndex") === userDict("userIdIndex"))
      .join(itemDict, indexedWithNegative("itemIdIndex") === itemDict("itemIdIndex"))
      .select(userDict("userIdIndex"), itemDict("itemIdIndex"), col("label"), col("userVec"), col("itemVec"))
      .withColumn("cosineSimilarity", getCosineSim(col("userVec"), col("itemVec")))
      .sort(col("cosineSimilarity").desc)
      .withColumnRenamed("cosineSimilarity", "score")

    joined.cache()

    joined.show(5)

    val precisionRecall = (0.1 to 0.9 by 0.1)
      .map(x => calculatePreRec(x.toFloat, joined))
      .sortBy(-_._1)

    precisionRecall.foreach(x => println(x.toString))

    val decile = toDecile(joined).orderBy(col("bucket").desc)
    decile.show

    val out = userDict.select("userIdIndex", "userVec").
      join(itemDict.select("itemIdIndex", "itemVec"))
      .withColumn("cosineSimilarity", getCosineSim(col("userVec"), col("itemVec")))
      .sort(col("cosineSimilarity").desc)

    // out.show()

    val time2 = System.nanoTime()
    val rankTime = (time2 - time1) * (1e-9)
    println("rankTime (s):  " + rankTime)
  }

}
