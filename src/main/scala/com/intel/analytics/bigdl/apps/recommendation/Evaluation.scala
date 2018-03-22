package com.intel.analytics.bigdl.apps.recommendation

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.DataFrame
import com.intel.analytics.bigdl.apps.recommendation.Utils._
import org.apache.spark.rdd.RDD

object Evaluation {

  def evaluate(evaluateDF: DataFrame) = {

    val binaryEva = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
    val out = binaryEva.evaluate(evaluateDF)
    println("AUROC: " + toDecimal(3)(out))

    val multiEva1 = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    val out1 = multiEva1.evaluate(evaluateDF)
    println("accuracy: " + toDecimal(3)(out1))


    val multiEva2 = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision")
    val out2 = multiEva2.evaluate(evaluateDF)
    println("precision: " + toDecimal(3)(out2))

    val multiEva3 = new MulticlassClassificationEvaluator().setMetricName("weightedRecall")
    val out3 = multiEva3.evaluate(evaluateDF)
    println("recall: " + toDecimal(3)(out3))

    Seq(out, out1, out2, out3).map(x => toDecimal(3)(x))
  }


}
