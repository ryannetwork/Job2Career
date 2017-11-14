package com.intel.analytics.bigdl.apps.recommendation

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.DataFrame
import com.intel.analytics.bigdl.apps.recommendation.Utils._
import org.apache.spark.rdd.RDD

object Evaluation {

  def evaluate(evaluateDF: DataFrame) = {
    val binaryEva = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
    val out1 = binaryEva.evaluate(evaluateDF)
    println("AUROC: " + toDecimal(3)(out1))

    val multiEva = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision")
    val out2 = multiEva.evaluate(evaluateDF)
    println("precision: " + toDecimal(3)(out2))

    val multiEva2 = new MulticlassClassificationEvaluator().setMetricName("weightedRecall")
    val out3 = multiEva2.evaluate(evaluateDF)
    println("recall: " + toDecimal(3)(out3))

    Seq(out1, out2, out3).map(x=> toDecimal(3)(x))
  }


}
