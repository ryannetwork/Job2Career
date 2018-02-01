# POC: Jobs recommendation on Apache Spark and BigDL

## What is BigDL?
[BigDL](https://github.com/intel-analytics/BigDL/) is a distributed deep learning library for Apache Spark; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters.

## Deploy BigDL application on Databricks cloud
1. Step1 Build jar at local
    * Mvn clean install assembly:assembly
2. Step 2 login on to web with credentials
3. Step 3 setup cluster
    * Clusters -> create cluster
    * give a name “intel” set up workers 1, uncheck auto scaling. 
    * Set up spark configuration here, for example
        * spark.executor.cores 4
        * spark.cores.max 4
        * spark.shuffle.reduceLocality.enabled false
        * spark.shuffle.blockTransferService nio
        * spark.scheduler.minRegisteredResourcesRatio 1.0
        * spark.speculation false
4. Step 4, upload data and dependency jar
    * Data-> create table -> upload data, give a name for example ”NEG50”
    * /FileStore/taAbles/Jobs2Career/indexed/indexed/
    * /FileStore/tables/Jobs2Career/indexed/NEG50/
    * /FileStore/tables/Jobs2Career/lib/job2career_1_0_SNAPSHOT_job-0ca74.jar
5. Step 5 run job
    * Jobs -> Create job -> give a name
    * set Jar, Upload jar, give main class “com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove”, give arguments "--inputDir /FileStore/tables/Jobs2Career/indexed/“
    * Add dependency lib dbfs:/FileStore/tables/Jobs2Career/lib/job2career_1_0_SNAPSHOT_job-0ca74.jar
    * Edit cluster -> existing cluster, choose the one you created 
    * confirm -> run now -> see results from log