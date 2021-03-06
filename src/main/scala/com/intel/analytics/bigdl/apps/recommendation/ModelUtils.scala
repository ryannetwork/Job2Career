package com.intel.analytics.bigdl.apps.recommendation

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor

case class ModelParam(userEmbed: Int = 20,
                      itemEmbed: Int = 20,
                      mfEmbed: Int = 20,
                      hiddenLayers: Array[Int] = Array(40, 20, 10),
                      labels: Int = 2) {
  override def toString: String = {
    "userEmbed =" + userEmbed + "\n" +
      " itemEmbed = " + itemEmbed + "\n" +
      " mfEmbed = " + mfEmbed + "\n" +
      " midLayer = " + hiddenLayers.mkString("|") + "\n" +
      " labels = " + labels
  }
}

class ModelUtils(modelParam: ModelParam) {

  def this() = {
    this(ModelParam())
  }

  def mlp(userCount: Int, itemCount: Int) = {

    println(modelParam)

    val input = Identity().inputs()
    val select1: ModuleNode[Float] = Select(2, 1).inputs(input)
    val select2: ModuleNode[Float] = Select(2, 2).inputs(input)

    val userTable = LookupTable(userCount, modelParam.userEmbed)
    val itemTable = LookupTable(itemCount, modelParam.itemEmbed)
    userTable.setWeightsBias(Array(Tensor[Float](userCount, modelParam.userEmbed).randn(0, 0.1)))
    itemTable.setWeightsBias(Array(Tensor[Float](itemCount, modelParam.itemEmbed).randn(0, 0.1)))

    val userTableInput: ModuleNode[Float] = userTable.inputs(select1)
    val itemTableInput = itemTable.inputs(select2)

    val embeddedLayer = JoinTable(2, 0).inputs(userTableInput, itemTableInput)

    val linear1: ModuleNode[Float] = Linear(modelParam.itemEmbed + modelParam.userEmbed,
      modelParam.hiddenLayers(0)).inputs(embeddedLayer)

    val midLayer = buildMlpModuleNode(linear1, 1, modelParam.hiddenLayers)

    val reluLast = ReLU().inputs(midLayer)
    val last: ModuleNode[Float] = Linear(modelParam.hiddenLayers.last, modelParam.labels).inputs(reluLast)

    val output = if (modelParam.labels >= 2) LogSoftMax().inputs(last) else Sigmoid().inputs(last)

    Graph(input, output)
  }


  def mlp2 = {

    println(modelParam)

    val input = Identity().inputs()

    // val linear1: ModuleNode[Float] = Linear(modelParam.itemEmbed + modelParam.userEmbed,
    val linear1: ModuleNode[Float] = Linear(100,
      40).inputs(input)

    val relu1 = ReLU().inputs(linear1)
    val linear2 = Linear(40, 20).inputs(relu1)

    val relu2 = ReLU().inputs(linear2)
    val linear3 = Linear(20, 10).inputs(relu2)

    val reluLast = ReLU().inputs(linear3)
    val last: ModuleNode[Float] = Linear(10, 2).inputs(reluLast)

    val output = if (modelParam.labels >= 2) LogSoftMax().inputs(last) else Sigmoid().inputs(last)

    Graph(input, output)
  }

  def mlpPreEmbed = {
    val featureSize = modelParam.userEmbed + modelParam.itemEmbed
    val hiddenLayers = modelParam.hiddenLayers
    val mlpModel = Sequential()
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
    mlpModel
  }

  def mlp3 = {
    val model = Sequential()
    model.add(Linear(100, 40, initWeight = Tensor(40, 100).randn(0, 0.1), initBias = Tensor(40).randn(0, 0.1)))
    model.add(ReLU())
    model.add(Linear(40, 20, initWeight = Tensor(20, 40).randn(0, 0.1), initBias = Tensor(20).randn(0, 0.1)))
    model.add(Linear(20, 10, initWeight = Tensor(10, 20).randn(0, 0.1), initBias = Tensor(10).randn(0, 0.1)))
    model.add(ReLU())
    model.add(Linear(10, 2, initWeight = Tensor(2, 10).randn(0, 0.1), initBias = Tensor(2).randn(0, 0.1)))
    model.add(ReLU())
    model.add(LogSoftMax())
    model
  }

  def mlp3_300 = {
    val model = Sequential()
    model.add(Linear(600, 40))
    model.add(ReLU())
    model.add(Linear(40, 20))
    model.add(ReLU())
    model.add(Linear(20, 10))
    model.add(ReLU())
    model.add(Linear(10, 2))
    model.add(ReLU())
    model.add(LogSoftMax())
    model
  }

  def mlp4_300 = {
    val model = Sequential()
    model.add(Linear(600, 100))
    model.add(ReLU())
    model.add(Linear(100, 40))
    model.add(ReLU())
    model.add(Linear(40, 20))
    model.add(ReLU())
    model.add(Linear(20, 10))
    model.add(ReLU())
    model.add(Linear(10, 2))
    model.add(ReLU())
    model.add(LogSoftMax())
    model
  }

  private def buildMlpModuleNode(linear: ModuleNode[Float], midLayerIndex: Int, midLayers: Array[Int]): ModuleNode[Float] = {

    if (midLayerIndex >= midLayers.length) {
      linear
    } else {
      val relu = ReLU().inputs(linear)
      val l = Linear(midLayers(midLayerIndex - 1), midLayers(midLayerIndex)).inputs(relu)
      buildMlpModuleNode(l, midLayerIndex + 1, midLayers)
    }

  }

  def ncf(userCount: Int, itemCount: Int) = {

    val mfUserTable = LookupTable(userCount, modelParam.mfEmbed)
    val mfItemTable = LookupTable(itemCount, modelParam.mfEmbed)
    val mlpUserTable = LookupTable(userCount, modelParam.userEmbed)
    val mlpItemTable = LookupTable(itemCount, modelParam.itemEmbed)

    val mfEmbeddedLayer = ConcatTable().add(Sequential().add(Select(2, 1)).add(mfUserTable))
      .add(Sequential().add(Select(2, 2)).add(mfItemTable))

    val mlpEmbeddedLayer = Concat(2).add(Sequential().add(Select(2, 1)).add(mlpUserTable))
      .add(Sequential().add(Select(2, 2)).add(mlpItemTable))

    val mfModel = Sequential()
    mfModel.add(mfEmbeddedLayer).add(CMulTable())

    val mlpModel = Sequential()
    mlpModel.add(mlpEmbeddedLayer)

    val linear1 = Linear(modelParam.itemEmbed + modelParam.userEmbed, modelParam.hiddenLayers(0))
    mlpModel.add(linear1).add(ReLU())

    for (i <- 1 to modelParam.hiddenLayers.length - 1) {
      mlpModel.add(Linear(modelParam.hiddenLayers(i - 1), modelParam.hiddenLayers(i))).add(ReLU())
    }

    val concatedModel = Concat(2).add(mfModel).add(mlpModel)

    val model = Sequential()
    model.add(concatedModel).add(Linear(modelParam.mfEmbed + modelParam.hiddenLayers.last, modelParam.labels))

    if (modelParam.labels >= 2) model.add(LogSoftMax()) else model.add(Sigmoid())
    model
  }

}
