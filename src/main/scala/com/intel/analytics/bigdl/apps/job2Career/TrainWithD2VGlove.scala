package com.intel.analytics.bigdl.apps.job2Career

import com.intel.analytics.bigdl.apps.recommendation.Utils._
import com.intel.analytics.bigdl.example.textclassification.TextClassifier.log
import com.intel.analytics.bigdl.example.utils.{AbstractTextClassificationParams, TextClassificationParams, TextClassifier}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import scopt.OptionParser

import scala.collection.immutable
import scala.io.Source

case class TrainParam(val inputDir: String = "/Users/guoqiong/intelWork/projects/jobs2Career/data/indexed_application_job_resume_2016_2017_10",
                      val outputDir: String = "/Users/guoqiong/intelWork/projects/jobs2Career/data/validation_predict",
                      val topK: Int = 500,
                      val dictDir: String = "/Users/guoqiong/intelWork/projects/wrapup/textClassification/keras/glove.6B/glove.6B.50d.txt",
                      val valDir: String = "/Users/guoqiong/intelWork/projects/jobs2Career/data/validation/part*",
                      val batchSize: Int = 1024,
                      val nEpochs: Int = 10,
                      val learningRate: Double = 1e-2,
                      val learningRateDecay: Double = 1e-5)

object TrainWithD2VGlove {

  def main(args: Array[String]): Unit = {

    val defaultParams = TrainParam()

    val parser = new OptionParser[TrainParam]("BigDL Example") {
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
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: TrainParam) = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[8]")
      .setAppName("app")

    val sc = SparkContext.getOrCreate(conf)
    val spark = SparkSession.builder().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val input = param.inputDir

    // val joined = DataProcess.negativeJoin(indexed, itemDict, userDict, br)
    //val joined = DataProcess.crossJoinAll(userDict, itemDict, br, indexed, 100)

    //val joined = spark.read.parquet(input + "/NEG50")

    val joined = spark.read.parquet(input + "/ALL")

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

    joined.printSchema()
    println("-------------abs rank dist----------------------------")
    getAbsRank(joined, param.topK).show(1000, false)

    //    rankDF.show()
    //
    //    val precisionRecall = getPrecisionRecall(joined)
    //    precisionRecall.foreach(x => println(x._1 + "," + x._2 + "," + x._3))
    //
    //    val buckets = bucketize(joined).orderBy(col("bucket").desc)
    //    buckets.show(100)
    //
    //
    //    Seq(3, 5, 10, 15, 20, 30).map(x => {
    //
    //      val (ratio, ndcg) = getHitRatioNDCG(joined, x)
    //      x + "," + ratio + "," + ndcg
    //
    //    }).foreach(println)

  }

  val gloveDir = s"/glove.6B/"

  val stopWordString = "a,about,above,after,again,against,all,am,an,and,any,are,as,at,be,because,been,before,being,below,between,both,but,by,could,did,do,does,doing,down,during,each,few,for,from,further,had,has,have,having,he,he’d,he’ll,he’s,her,here,here’s,hers,herself,him,himself,his,how,how’s,I,I’d,I’ll,I’m,I’ve,if,in,into,is,it,it’s,its,itself,let’s,me,more,most,my,myself,nor,of,on,once,only,or,other,ought,our,ours,ourselves,out,over,own,same,she,she’d,she’ll,she’s,should,so,some,such,than,that,that’s,the,their,theirs,them,themselves,then,there,there’s,these,they,they’d,they’ll,they’re,they’ve,this,those,through,to,too,under,until,up,very,was,we,we’d,we’ll,we’re,we’ve,were,what,what’s,when,when’s,where,where’s,which,while,who,who’s,whom,why,why’s,with,would,you,you’d,you’ll,you’re,you’ve,your,yours,yourself,yourselves"
  val stopWordSet = stopWordString.split(",") ++ Set(",", ".")

  def loadWordVecMap(filename: String): Map[String, Array[Float]] = {

    val sc = SparkContext.getOrCreate()
    val dict = sc.textFile(filename)
    // val wordMap = for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) yield {
    val wordMap = dict.map { line =>
      val values = line.split(" ")
      val word = values(0)
      val coefs: Array[Float] = values.slice(1, values.length).map(_.toFloat)
      (word, coefs)
    }
    wordMap.filter(x => !stopWordSet.contains(x._1)).collect().toMap
  }

  def doc2VecFromWordMap(dataFrame: DataFrame, brMap: Broadcast[Map[String, Array[Float]]], newCol: String, colName: String) = {

    val createVectorUdf = udf((doc: String) => {
      /**
        * x y z x
        **/
      val seq = doc.split("\n").flatMap(x => x.split(" ")).filter(x => x.size > 1)

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

    dataFrame.withColumn(newCol, createVectorUdf(col(colName))).drop(colName)
  }

}

