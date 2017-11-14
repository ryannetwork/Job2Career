package com.intel.analytics.bigdl.apps.job2Career

import com.intel.analytics.bigdl.apps.job2Career.DataProcess._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object DataAnalysis {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")


    val input2 = "/Users/guoqiong/intelWork/projects/jobs2Career/resume_search"
    val data = spark.read.parquet(input2 + "/application_job_resume_2016_2017_10.parquet")

    data.printSchema()
    data.cache()

    println("---------------------- job analysis --------------------------")
    val job = data.select("jobs_job_id", "jobs_title", "jobs_description")
      .filter(col("jobs_job_id").isNotNull && col("jobs_title").isNotNull && col("jobs_description").isNotNull)
    // job.show(2, false)

    println("distinct count job:" + job.distinct().count())
    println("distinct count job id:" + job.select("jobs_job_id").distinct().count())
    println("distinct count job_id title:" + job.select("jobs_job_id", "jobs_title").distinct().count())
    println("distinct count job_id description:" + job.select("jobs_job_id", "jobs_description").distinct().count())


    println("---------------------- resume analysis --------------------------")
    val resume = data.select("resume_url", "resume").withColumn("resume_id", createResumeId(col("resume_url")))
      .filter((col("resume.resume.normalizedBody").isNotNull))
      .filter(col("resume_id").isNotNull)

    println("distinct count resume_url:" + resume.select("resume_url").distinct().count())
    println("distinct count resume_id:" + resume.select("resume_id").distinct().count())
    println("distinct count resume_id with descreption:" + resume.select("resume_id", "resume.resume.normalizedBody").filter((col("resume.resume.normalizedBody").isNotNull))
      .distinct().count())


    println("---------------------- application analysis --------------------------")

    val application = data.withColumn("resume_id", createResumeId(col("resume_url"))).filter(col("resume_id").isNotNull)
    //application.show(2, false)
    println(application.count())

    val jobCountVsResumeCountDist: Array[(Long, Long)] = application
      .select("resume_id", "jobs_job_id").filter(col("resume_id").isNotNull).distinct.groupBy("resume_id").count.orderBy(desc("count"))
      .withColumnRenamed("count", "job_count")
      .groupBy("job_count").count.withColumnRenamed("count", "resume_count")
      .rdd.map(row => (row.getAs[Long](0), row.getAs[Long](1))).collect()

    println("application resume_url distinct count:" + application.select("resume_url").distinct().count())
    println("application job_id distinct count:" + application.select("jobs_job_id").distinct().count())
    println("application resume_id distinct count:" + application.select("resume_id").distinct().count())
    println("application status count:" + application.select("resume_id", "jobs_job_id").distinct().count())

    println("job_count,resume_count")
    jobCountVsResumeCountDist.sortBy(x => x._1).map(x => x._1 + "," + x._2).foreach(println)

    val realApplication = application.withColumn("label", applicationStatusUdf(col("new_application")))
      .select("resume_id", "jobs_job_id", "label")
      .filter(col("resume_id").isNotNull && col("jobs_job_id").isNotNull && col("label").isNotNull)

    println("real  application count :" + realApplication.count())

    val applicationWithContent = application.withColumn("label", applicationStatusUdf(col("new_application")))
      .select("resume_id", "jobs_job_id", "label", "resume.resume.normalizedBody", "jobs_description")
      .filter(col("resume_id").isNotNull && col("jobs_job_id").isNotNull && col("label").isNotNull)
      .filter(col("resume.resume.normalizedBody").isNotNull && col("jobs_description").isNotNull)
      .distinct()

    println("real  application with content count :" + applicationWithContent.count())

    println("total counts: " + data.count())
  }


}
