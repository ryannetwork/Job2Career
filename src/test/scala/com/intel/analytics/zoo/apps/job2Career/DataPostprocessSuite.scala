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
    val aprData ="/home/arda/intelWork/projects/jobs2Career/data/validation-with-click_2018-04-01_2018-04-30_click-filter"
    val oldVali = "/home/arda/intelWork/projects/jobs2Career/data/validation"
    val df1 = spark.read.parquet(aprData)
    df1.printSchema()
    println(df1.count())
    df1.groupBy("apply_flag", "is_clicked").count().show()

    val preprocessedData = "/home/arda/intelWork/projects/jobs2Career/preprocessed/"
    val indexed = spark.read.parquet(preprocessedData + "indexed")
    val userDict = spark.read.parquet(preprocessedData + "/userDict")
    val itemDict = spark.read.parquet(preprocessedData + "/itemDict")
    indexed.show(5)
    userDict.show(5)
    itemDict.show(5)
    println(indexed.count())
    println(userDict.count())
    println(itemDict.count())
    indexed.groupBy("label").count().show()
  }
}
