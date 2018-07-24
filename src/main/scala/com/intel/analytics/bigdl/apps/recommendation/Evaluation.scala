package com.intel.analytics.bigdl.apps.recommendation

import com.intel.analytics.bigdl.apps.job2Career.{DataProcess, Utils}
import com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove.loadWordVecMap
import com.intel.analytics.bigdl.apps.job2Career.TrainWithEnsambleNCF_Glove.{ncfPredict, ncfWithKmeansPredict}
import com.intel.analytics.bigdl.apps.job2Career.Utils.AppParams
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.{DataFrame, SQLContext}
import com.intel.analytics.bigdl.apps.recommendation.Utils._
import com.intel.analytics.zoo.models.common.ZooModel
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

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

  def evaluate2(evaluateDF: DataFrame) = {
    val truePositive = evaluateDF.filter(col("prediction") === 1.0 && col("label") === 1.0).count()
    val falsePositive = evaluateDF.filter(col("prediction") === 1.0 && col("label") === 0.0).count()
    val trueNegative = evaluateDF.filter(col("prediction") === 0.0 && col("label") === 0.0).count()
    val falseNegative = evaluateDF.filter(col("prediction") === 0.0 && col("label") === 1.0).count()

    val accuracy_denominator = trueNegative.toDouble + truePositive.toDouble + falseNegative.toDouble + falsePositive.toDouble
    val precision_denominator = truePositive.toDouble + falsePositive.toDouble
    val recall_denomiator = truePositive.toDouble + falseNegative.toDouble
    // handle scenario when denominator became 0. Check log for more detailed info to debug
    val accuracy = if (accuracy_denominator > 0) (truePositive.toDouble + trueNegative.toDouble) / accuracy_denominator else -1
    val precision = if (precision_denominator > 0) truePositive.toDouble / precision_denominator else -1
    val recall = if (recall_denomiator > 0) truePositive.toDouble / recall_denomiator else -1

    println("truePositive: " + truePositive)
    println("falsePositive: " + falsePositive)
    println("trueNegative: " + trueNegative)
    println("falseNegative: " + falseNegative)
    println("accuracy: " + accuracy)
    println("precision: " + precision)
    println("recall: " + recall)

    val evaluation = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
    val evaluatLabels = evaluateDF
      .withColumn("label", col("label").cast("double"))
      .withColumn("prediction", col("prediction").cast("double"))
    val modelAUROC = evaluation.setMetricName("areaUnderROC").evaluate(evaluatLabels)
    val modelAUPR = evaluation.setMetricName("areaUnderPR").evaluate(evaluatLabels)
    println("modelAUROC: " + modelAUROC)
    println("modelAUPR: " + modelAUPR)

    //Seq(accuracy, precision, recall, modelAUROC, modelAUPR)
    Seq(accuracy, precision, recall, modelAUROC, modelAUPR).map(x => toDecimal(3)(x))

  }

  def evaluatePredictions(predictionsIn: DataFrame) = {
    val predictions = predictionsIn.withColumn("label", toZero(col("label")))
      .withColumn("prediction", toZero(col("prediction")))

    predictions.persist(StorageLevel.DISK_ONLY)
    val metrics = Evaluation.evaluate2(predictions)
    predictions.unpersist()
    metrics
  }

}
