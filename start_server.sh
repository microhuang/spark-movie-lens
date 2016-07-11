export SPARK_MOVIE_LENS_DIR="hdfs:/tmp"

SPARK_HOME=~/spark-2.0.0-preview-bin-hadoop2.6

#SPARK_MASTER=spark://169.254.206.2:7077
#SPARK_MASTER=spark://192.168.0.11:7077
SPARK_MASTER=yarn

${SPARK_HOME}/bin/spark-submit --master ${SPARK_MASTER} --deploy-mode client --queue default server.py 
