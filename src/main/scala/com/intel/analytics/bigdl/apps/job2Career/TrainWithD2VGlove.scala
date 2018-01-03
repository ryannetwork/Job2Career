package com.intel.analytics.bigdl.apps.job2Career

import com.intel.analytics.bigdl.apps.recommendation.Utils._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

import scala.collection.immutable
import scala.io.Source


object TrainWithD2VGlove {

  val gloveDir = s"/glove.6B/"

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


  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[8]")
      .setAppName("app")

    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    // val input = "/Users/guoqiong/intelWork/projects/jobs2Career/data/"
    val input = "/Users/guoqiong/intelWork/projects/jobs2Career/data/indexed_application_job_resume_2016_2017_10"

    // val joined = DataProcess.negativeJoin(indexed, itemDict, userDict, br)
    // val joined = DataProcess.crossJoinAll(userDict, itemDict, br, indexed, 100)

    val joined = spark.read.parquet(input + "/NEG")
    //val joined = spark.read.parquet(input + "/ALL")

    joined.filter("label = 1").groupBy("userIdIndex").count()
      .withColumnRenamed("count", "applyJobCount")
      .groupBy("applyJobCount").count()
      .orderBy("applyJobCount").show(1000, false)

    joined.filter("label = 0").groupBy("userIdIndex").count()
      .withColumnRenamed("count", "notApplyJobCount")
      .groupBy("notApplyJobCount").count()
      .orderBy("notApplyJobCount").show(1000, false)

    joined.cache()

    joined.show(5)

    val precisionRecall = getPrecisionRecall(joined)
    precisionRecall.foreach(x => println(x._1 + "," + x._2 + "," + x._3))

    val buckets = bucketize(joined).orderBy(col("bucket").desc)
    buckets.show(100)

    Seq(3, 5, 10, 15, 20, 30).map(x => {

      val (ratio, ndcg) = getHitRatioNDCG(joined, x)
      x + "," + ratio + "," + ndcg

    }).foreach(println)

  }

}
