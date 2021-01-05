import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

object StocksPredicting extends App {

  /** Hides red INFO lines */
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val spark = SparkSession.builder()
    .appName("Final Project")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

  /** Convert date to the right format */
  def toDate(col: org.apache.spark.sql.Column, formats: Seq[String]): org.apache.spark.sql.Column =
    coalesce(formats.map(fmt => to_date(col, fmt)): _*)
  val dateSeq = Seq("MMM dd yyyy", "yyyy-MM-dd", "dd-MMM-yy")


  /** DataFrames paths */
  val givenStocks = "src/main/resources/stock_prices.csv"
  val stocks2 = "src/main/resources/stocks3.csv"
  val stocks3 = "src/main/resources/stocks5.csv"


  val stocks: DataFrame = spark.read.option("header", "true").csv(stocks2)
    .select(
      unix_timestamp(toDate(col("date"), dateSeq)).as("date"),
      col("ticker"),
      col("close").cast(DoubleType).as("close"),
     )
  stocks.cache()

  /** Transforms data for the model and fits it */
  val tickerIndexer = new StringIndexer()
    .setInputCol("ticker").setOutputCol("IndTicker")
    .setHandleInvalid("keep")

  val encoder = new OneHotEncoder()
    .setInputCol(tickerIndexer.getOutputCol).setOutputCol("tickerInd")

  val assembler = new VectorAssembler()
    .setInputCols(Array("date", "tickerInd"))
    .setOutputCol("features")

  val pipeline = new Pipeline().setStages(Array(tickerIndexer, encoder, assembler))

  val trainTestDF = pipeline.fit(stocks).transform(stocks)
    .withColumnRenamed("close", "label")

  val Array(train, test) = trainTestDF.select("date", "ticker", "IndTicker", "features", "label")
    .randomSplit(Array(0.8, 0.2))

  val lr = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
  val lrModel = lr.fit(train)

  println("LINEAR REGRESSION MODEL")
  lrModel.transform(test).show(false)


  /** Prints the summary */
  val trainingSummary = lrModel.summary
  println(s"Coefficients: ${lrModel.coefficients} \nIntercept: ${lrModel.intercept}")
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}\n")

  /** Creates new data for predicting */
  val numberOfTickers = stocks.select("ticker").distinct().count()

  val newData: DataFrame = numberOfTickers match {
    case 3 => spark.createDataFrame(Seq(
      ("19-Jan-17", "IBM"), ("19-Jan-17", "AMZN"), ("19-Jan-17", "YHOO"),
      ("20-Jan-17", "IBM"), ("20-Jan-17", "AMZN"), ("20-Jan-17", "YHOO")))
    case 5 => spark.createDataFrame(Seq(
      ("2016-11-03", "AAPL"), ("2016-11-03", "MSFT"), ("2016-11-03", "TSLA"),  ("2016-11-03", "GOOG"), ("2016-11-03", "BLK"),
      ("2016-11-04", "AAPL"), ("2016-11-04", "MSFT"), ("2016-11-04", "TSLA"), ("2016-11-04", "GOOG"), ("2016-11-04", "BLK")))
  }
  val newDF = newData.toDF("date", "ticker")
    .withColumn("date", unix_timestamp(toDate(col("date"), dateSeq)))


  val predictDF = pipeline.fit(newDF).transform(newDF)
  val predictions = lrModel.transform(predictDF).select("date", "ticker", "prediction")
  predictions.show()

  spark.close()
}
