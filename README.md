# POC: Jobs recommendation on Apache Spark and BigDL

## What is BigDL?
[BigDL](https://github.com/intel-analytics/BigDL/) is a distributed deep learning library for Apache Spark; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters.
* **Rich deep learning support.** Modeled after [Torch](http://torch.ch/), BigDL provides comprehensive support for deep learning, including numeric computing (via [Tensor](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor)) and high level [neural networks](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/nn); in addition, users can load pre-trained [Caffe](http://caffe.berkeleyvision.org/) or [Torch](http://torch.ch/) models into Spark programs using BigDL.


## Deploy on Databricks cloud
1. Step1 Build jar at local
    * Mvn clean install assembly:assembly
2. Step 2 login on to web with credentials
3. Step 3 setup cluster
    * Clusters -> new cluster
    * give a name “intel-bvigdl” set up workers 1, uncheck auto scaling. 
4. Step 4, upload data
    * Data-> create table -> upload data, give a name for example”NEG50", and remember the path uploaded"  /FileStore/tables/NEG50/part_00000_fad5a818_adda_4bc3_8821_5f5e55737039_c000_snappy-14be5.parquet
5. Step 5 run job
    * Jobs -> Create job -> give a name -> set Jar
    * Upload jar, give main class “com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove”, give arguments "--inputDir /FileStore/tables/“
    * Edit cluster -> existing cluster, choose the one you created -> confirm -> run now -> see results from log
