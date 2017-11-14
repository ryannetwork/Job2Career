~/intelWork/tools/spark-2.1.1-bin-hadoop2.7/bin/spark-submit \
--master local[8] \
--driver-memory 16g \
--jars ./target/application-1.0-SNAPSHOT-job.jar \
--executor-memory 16g \
--class com.intel.analytics.bigdl.apps.job2Career.TrainWithNCF \
./target/application-1.0-SNAPSHOT.jar
#--class com.intel.analytics.bigdl.apps.job2Career.DataAnalysis \
#--class com.intel.analytics.bigdl.apps.job2Career.DataProcess \
#--class com.intel.analytics.bigdl.apps.job2Career.TrainWithALS \
#--class com.intel.analytics.bigdl.apps.job2Career.TrainWithD2VGlove \
