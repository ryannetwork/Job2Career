~/intelWork/tools/spark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit \
--master local[8] \
--driver-memory 16g \
--jars target/job2Career-1.0-SNAPSHOT-job.jar \
--executor-memory 16g \
--class com.intel.analytics.bigdl.apps.job2Career.TrainWithNCF_Glove \
target/job2career-1.0-SNAPSHOT.jar \
--batchSize 1000 \
--nEpochs 10  \
--lRate 0.001 
#--class com.intel.analytics.bigdl.apps.job2Career.DataAnalysis \
#--class com.intel.analytics.bigdl.apps.job2Career.DataProcess \
#--class com.intel.analytics.bigdl.apps.job2Career.TrainWithALS \
#--class com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove \
