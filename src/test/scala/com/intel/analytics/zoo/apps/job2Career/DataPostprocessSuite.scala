package com.intel.analytics.zoo.apps.job2Career

import com.intel.analytics.bigdl.apps.job2Career.TrainWithEnsambleNCF_Glove
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{Dataset, SparkSession}
import org.scalatest.{BeforeAndAfter, FunSuite}

case class User(id: Int, score: Double)

case class User2(id: Int, feature: Array[Double])

class DataPostprocessSuite extends FunSuite with BeforeAndAfter {

  var spark: SparkSession = _

  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("Train")
      .setMaster("local[8]")
    spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
  }

  test("metrics") {
    val sparkSession = SparkSession.builder().getOrCreate()
    import sparkSession.implicits._
    val data = Array((0, 0.1), (1, 0.8), (2, 0.2)).map(x => User(x._1, x._2))
    val dataset = spark.createDataset(data)
    dataset.show(10)
    dataset.select("id").show()

    dataset.map(x => x.score + 1).show()
  }


  test("Kmean") {

    val sparkSession = SparkSession.builder().getOrCreate()
    import sparkSession.implicits._
    val data = Array((0, Array(0.1, 0.2)), (1, Array(0.8, 0.9)), (2, Array(0.15, 0.2)), (3, Array(0.7, 0.9))).map(x => User2(x._1, x._2))
    val dataset: Dataset[User2] = spark.createDataset(data)

  }

  test("new data") {
    val df = spark.read.parquet("/home/arda/intelWork/projects/jobs2Career/data/validation_2018-03-01_2018-03-31_click-filter")
    df.show()
    df.printSchema()
  }
}
