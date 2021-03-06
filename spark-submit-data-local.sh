#!/bin/bash
/home/arda/intelWork/tools/spark/spark-2.3.1-bin-hadoop2.7/bin/spark-submit \
--master local[16] \
--class com.intel.analytics.bigdl.apps.job2Career.TrainWithEnsambleNCF_Glove \
--executor-memory 16G \
--num-executors 2 \
--executor-cores 8 \
--driver-memory 10g \
--conf spark.sql.shuffle.partitions=1000 \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.ui.showConsoleProgress=false \
--conf spark.yarn.max.executor.failures=4 \
--conf spark.yarn.executor.memoryOverhead=4096 \
--conf spark.yarn.driver.memoryOverhead=10240 \
--conf spark.sql.tungsten.enabled=true \
--conf spark.locality.wait=1s \
--conf spark.yarn.maxAppAttempts=4 \
--conf spark.driver.extraJavaOptions="-XX:ParallelGCThreads=8 -XX:+UseParallelGC -XX:+UseParallelOldGC -Dderby.system.home=/tmp/uc/derby" \
--conf spark.executor.extraJavaOptions="-XX:ParallelGCThreads=8 -XX:+UseParallelGC -XX:+UseParallelOldGC" \
--conf spark.serializer=org.apache.spark.serializer.JavaSerializer \
--conf spark.driver.maxResultSize=10g \
--conf spark.sql.parquet.binaryAsString=true \
--conf spark.rdd.compress=true \
 dist/job2career-with-dependencies.jar \
--dataPathParams  "/home/arda/intelWork/projects/jobs2Career/resume_search/application_job_resume_2016_2017_10.parquet,/home/arda/intelWork/projects/jobs2Career/preprocessed/,/home/arda/intelWork/projects/jobs2Career/modelOutput/,/home/arda/intelWork/projects/jobs2Career/data/validation/part*,true" \
--ncfParams "1024, 10, 1e-2, 1e-5, true, /home/arda/intelWork/projects/jobs2Career/model/ncfWithKmeans"  \
--kmeansParams "3, 20, true, userVec, /home/arda/intelWork/projects/jobs2Career/model/kmeans" \
--gloveParams "/home/arda/intelWork/projects/wrapup/textClassification/keras/glove.6B/glove.6B.50d.txt, 50" \
--negativeK 1 \
--mode "NCFWithKmeans"
#--mode "data"
