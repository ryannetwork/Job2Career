package com.intel.analytics.bigdl.apps.recommendation

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.recommendation.NeuralCF
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.zoo.models.common.ZooModel

import scala.reflect.ClassTag

class NCFJob2Career(override val userCount: Int,
                    override val itemCount: Int,
                    override val numClasses: Int = 2,
                    override val userEmbed: Int = 50,
                    override val itemEmbed: Int = 50,
                    override val hiddenLayers: Array[Int] = Array(40, 20, 10),
                    override val includeMF: Boolean = true,
                    override val mfEmbed: Int = 20,
                    val trainEmbed: Boolean = false)
                   (implicit ev: TensorNumeric[Float]) extends NeuralCF[Float](
  userCount, itemCount, numClasses, userEmbed, itemEmbed, hiddenLayers, includeMF, mfEmbed) {

  override def buildModel(): AbstractModule[Tensor[Float], Tensor[Float], Float] = {

    val model = Sequential[Float]()

    val mlpModel = Sequential[Float]()
    if (trainEmbed) {
      val mlpUserTable = LookupTable[Float](userCount, userEmbed)
      val mlpItemTable = LookupTable[Float](itemCount, itemEmbed)
      mlpUserTable.setWeightsBias(Array(Tensor[Float](userCount, userEmbed).randn(0, 0.1)))
      mlpItemTable.setWeightsBias(Array(Tensor[Float](itemCount, itemEmbed).randn(0, 0.1)))
      val mlpEmbeddedLayer = Concat[Float](2)
        .add(Sequential[Float]().add(Select(2, 1)).add(mlpUserTable))
        .add(Sequential[Float]().add(Select(2, 2)).add(mlpItemTable))
      mlpModel.add(mlpEmbeddedLayer)
    }
    val featureSize = userEmbed + itemEmbed
    val linear1 = Linear(featureSize, hiddenLayers(0),
      initWeight = Tensor(hiddenLayers(0), featureSize).randn(0, 0.1),
      initBias = Tensor(hiddenLayers(0)).randn(0, 0.1))
    mlpModel.add(linear1).add(ReLU())
    for (i <- 1 to hiddenLayers.length - 1) {
      mlpModel.add(
        Linear(hiddenLayers(i - 1), hiddenLayers(i),
          initWeight = Tensor(hiddenLayers(i), hiddenLayers(i - 1)).randn(0, 0.1),
          initBias = Tensor(hiddenLayers(i)).randn(0, 0.1))
      ).add(ReLU())
    }

    if (includeMF) {
      require(mfEmbed > 0, s"please provide meaningful number of embedding units")
      val mfUserTable: LookupTable[Float] = LookupTable[Float](userCount, mfEmbed)
      val mfItemTable = LookupTable[Float](itemCount, mfEmbed)
      mfUserTable.setWeightsBias(Array(Tensor[Float](userCount, mfEmbed).randn(0, 0.1)))
      mfItemTable.setWeightsBias(Array(Tensor[Float](itemCount, mfEmbed).randn(0, 0.1)))
      val mfEmbeddedLayer = ConcatTable()
        .add(Sequential[Float]().add(Select(2, 1)).add(mfUserTable))
        .add(Sequential[Float]().add(Select(2, 2)).add(mfItemTable))
      val mfModel = Sequential[Float]()
      mfModel.add(mfEmbeddedLayer).add(CMulTable())
      val concatedModel = Concat(2).add(mfModel).add(mlpModel)
      model.add(concatedModel)
        .add(Linear(mfEmbed + hiddenLayers.last, numClasses))
    }
    else {
      model.add(mlpModel).
        add(Linear(hiddenLayers.last, numClasses))
    }
    model.add(LogSoftMax[Float]())

    model.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]]
  }

}


object NCFJob2Career {
  /**
    * The factory method to create a NeuralCF instance.
    */
  def apply(
             userCount: Int,
             itemCount: Int,
             numClasses: Int,
             userEmbed: Int = 20,
             itemEmbed: Int = 20,
             hiddenLayers: Array[Int] = Array(40, 20, 10),
             includeMF: Boolean = true,
             mfEmbed: Int = 20,
             trainEmbed: Boolean = false)(implicit ev: TensorNumeric[Float]): NCFJob2Career = {
    new NCFJob2Career(userCount, itemCount, numClasses, userEmbed,
      itemEmbed, hiddenLayers, includeMF, mfEmbed, trainEmbed).build()
  }

  /**
    * Load an existing NeuralCF model (with weights).
    *
    * @param path       The path for the pre-defined model.
    *                   Local file system, HDFS and Amazon S3 are supported.
    *                   HDFS path should be like "hdfs://[host]:[port]/xxx".
    *                   Amazon S3 path should be like "s3a://bucket/xxx".
    * @param weightPath The path for pre-trained weights if any. Default is null.
    * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
    */
  def loadModel[T: ClassTag](
                              path: String,
                              weightPath: String = null)(implicit ev: TensorNumeric[T]): NCFJob2Career = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[NCFJob2Career]
  }
}

