package com.intel.analytics.bigdl.apps.job2Career

import scopt.OptionParser

object Utils {

  val Mode_Data = "data"
  val Mode_NCF = "NCFOnly"
  val Mode_NCFWithKeans = "NCFWithKmeans"
  val seperator = ","

  case class GloveParam(dictDir: String,
                        vectDim: Int)

  case class KmeansParam(kClusters: Int,
                         numIterations: Int,
                         isTrain: Boolean,
                         featureVec: String,
                         modelPath: String)

  case class NCFParam(batchSize: Int,
                      nEpochs: Int,
                      learningRate: Double,
                      learningRateDecay: Double,
                      isTrain: Boolean,
                      modelPath: String)

  case class DataParam(rawDir: String,
                       preprocessedDir: String,
                       modelOutput: String,
                       evaluateDir: String,
                       labelCol: String)

  case class AppParams(dataPathParams: DataParam = DataParam("/home/arda/intelWork/projects/jobs2Career/data/validation-with-click_2018-*", "/home/arda/intelWork/projects/jobs2Career/preprocessed/", "/home/arda/intelWork/projects/jobs2Career/modelOutput/", "/home/arda/intelWork/projects/jobs2Career/data/validation-with-click_2018-03-01_2018-03-31_click-filter", "is_clicked"),
                       //       ncfParams: NCFParam = NCFParam(1024, 10, 5e-3, 1e-6, true, "/Users/guoqiong/intelWork/projects/jobs2Career/model/ncf"),
                       ncfParams: NCFParam = NCFParam(1024, 20, 5e-2, 1e-6, true, "/home/arda/intelWork/projects/jobs2Career/model/ncfWithKmeans"),
                       kmeansParams: KmeansParam = KmeansParam(3, 20, true, "userVec", "/home/arda/intelWork/projects/jobs2Career/model/kmeans"),
                       gloveParams: GloveParam = GloveParam("/home/arda/intelWork/projects/wrapup/textClassification/keras/glove.6B/glove.6B.50d.txt", 50),
                       defaultPartition: Int = 60,
                       negativeK: Int = 1,
                       mode: String = Mode_NCF)

  val trainParser = new OptionParser[AppParams]("Recommendation demo") {
    head("AppParams:")

    opt[String]("dataPathParams")
      .text("dataPath Params")
      .action((x, c) => {
        val pArr = x.split(seperator).map(_.trim)
        val p = DataParam(pArr(0), pArr(1), pArr(2), pArr(3), pArr(4))
        c.copy(dataPathParams = p)
      })
    opt[String]("ncfParams")
      .text("ncfParams")
      .action((x, c) => {
        val pArr = x.split(seperator).map(_.trim)
        val p = NCFParam(pArr(0).toInt, pArr(1).toInt, pArr(2).toDouble, pArr(3).toDouble,
          pArr(4).toBoolean, pArr(5))
        c.copy(ncfParams = p)
      })
    opt[String]("kmeansParams")
      .text("kmeansParams")
      .action((x, c) => {
        val pArr = x.split(seperator).map(_.trim)
        val p = KmeansParam(pArr(0).toInt, pArr(1).toInt, pArr(2).toBoolean, pArr(3), pArr(4))
        c.copy(kmeansParams = p)
      })
    opt[String]("gloveParams")
      .text("gloveParams")
      .action((x, c) => {
        val pArr = x.split(seperator).map(_.trim)
        val p = GloveParam(pArr(0), pArr(1).toInt)
        c.copy(gloveParams = p)
      })
    opt[String]("negativeK")
      .text("negativeK")
      .action((x, c) => c.copy(negativeK = x.toInt))
    opt[String]("mode")
      .text("mode")
      .action((x, c) => c.copy(mode = x))
  }

}
