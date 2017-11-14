package com.intel.analytics.bigdl.apps.job2Career

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql._

import scala.collection.immutable
import scala.io.Source
import com.intel.analytics.bigdl.apps.job2Career.DataProcess._
import org.apache.log4j.{Level, Logger}


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

    //TODO choose one of the descriptions for both user and item

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

    (indexed, userDict, itemDict)
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val dataPath = "/Users/guoqiong/intelWork/projects/jobs2Career/"
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

    indexed.write.mode(SaveMode.Overwrite).parquet(output + "/indexed")
    userDict.write.mode(SaveMode.Overwrite).parquet(output + "/userDict")
    itemDict.write.mode(SaveMode.Overwrite).parquet(output + "/itemDict")
  }


}
