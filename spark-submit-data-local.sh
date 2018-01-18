#!/bin/bash
/Users/guoqiong/intelWork/tools/spark/spark-2.2.0-bin-hadoop2.7/bin/spark-submit \
--master local[8] \
--jars target/job2career-1.0-SNAPSHOT-job.jar \
--class com.intel.analytics.bigdl.apps.job2Career.TrainWithNCF \
--executor-memory 4G \
--num-executors 2 \
--executor-cores 4 \
--driver-memory 50g \
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
target/job2career-1.0-SNAPSHOT.jar \
