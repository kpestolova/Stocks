
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.LongType
import org.apache.log4j.Logger
import org.apache.log4j.Level



object StocksAnalyzing extends App {

  /** Hides red INFO lines */
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  /** DataFrames paths */
  val givenStocks = "src/main/resources/stock_prices.csv"
  val stocks2 = "src/main/resources/stocks3.csv" // doesn't have column "volume"
  val stocks3 = "src/main/resources/stocks5.csv"

  val spark = SparkSession.builder()
    .appName("Final Project")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

  /** Convert date to the right format */
  def toDate(col: org.apache.spark.sql.Column, formats: Seq[String]): org.apache.spark.sql.Column =
    coalesce(formats.map(fmt => to_date(col, fmt)): _*)
  val dateSeq = Seq("MMM dd yyyy", "yyyy-MM-dd", "dd-MMM-yy")

  val stocks: DataFrame = spark.read.option("header", "true").csv(givenStocks)
    .withColumn("date", toDate(col("date"), dateSeq))
  stocks.cache()

  /** Returns Average Daily Return of all stocks in percents
   * counted by formula:
   *
   * (closing price / previous day's closing price -1) * 100 */
  val w = Window.partitionBy("ticker").orderBy("date")

  val dailyReturn =  stocks.withColumn("Daily Return, %",
    (col("close") / lag("close", 1).over(w) - 1) * 100)
  dailyReturn.cache()

  val averageReturnAll = dailyReturn.groupBy("date")
    .agg(avg("Daily Return, %").as("Average  Daily Return"))

  val averageReturnEach = dailyReturn
    .groupBy("ticker")
    .agg(
      avg("Daily Return, %").as("Average  Daily Return"),
      stddev("Daily Return, %").as("Standard Deviation of Daily Returns"))

  println("AVERAGE DAILY RETURN OF ALL STOCKS")
  averageReturnAll.show()


/** Saves a DataFrame to a file of needed format */
  def saving(df: DataFrame, format: String): Unit = df.orderBy("date")
    .repartition(1).write
    .format(format)
    .mode(SaveMode.Overwrite)
    .option("header", "true")
    .save(s"src/main/resources/data/average_daily_return.$format")
   saving(averageReturnAll, "csv")

  /** Returns stocks Trading Frequency counted by:
   *
   * closing price * volume */
  val mostFrequently = stocks
    .withColumn("trade", col("close") * col("volume"))
    .groupBy("ticker")
    .agg(avg("trade").cast(LongType).as("Average Trade Frequency"))
    .orderBy(desc("Average Trade Frequency")).cache()

  println("STOCKS TRADING FREQUENCY")
  mostFrequently.select("ticker", "Average Trade Frequency").show(false)


  /** Returns Annualized Volatility counted by formula:
   *
   * Standard Deviation of Daily Returns * Square Root (Number of days)*/
  val days = stocks.select("date").distinct().count()

  val astddev = averageReturnEach.withColumn(
      "Annualized Standard Deviation", col("Standard Deviation of Daily Returns") * math.sqrt(days))
    .select("*")

  println("ANNUALIZED VOLATILITY")
  astddev.orderBy(desc("Annualized Standard Deviation")).cache().show(false)


  /** Returns Inc. Industries  */
  def getIndustryData = udf((review: String) =>
    review match {
      case "AAPL" => "Consumer Electronics"
      case "BLK"  => "Asset Management"
      case "GOOG" => "Internet Content & Information"
      case "MSFT" => "Software—Infrastructure"
      case "TSLA" => "Auto Manufacturers"
      case "IBM"  => "Information Technology Services"
      case "AMZN" => "Internet Retail"
      case "YHOO" => "Computer processing & cloud services"
    }
  )
  val industryData = averageReturnEach
    .drop("Standard Deviation of Daily Returns")
    .withColumn("Industry", getIndustryData(col("ticker")))
  industryData.orderBy(desc("Average  Daily Return")).show(false)

  spark.close()
}
