package com.sirolf2009.dl4j.rnn

import java.io.File
import java.nio.file.Files
import java.time.Duration
import org.influxdb.InfluxDB
import org.influxdb.InfluxDBFactory
import org.influxdb.dto.Query
import org.influxdb.dto.QueryResult
import org.influxdb.dto.QueryResult.Series

class Database {

	static val tradesDB = "tradesBTCUSD"

	val InfluxDB influxDB

	new(String host) {
		influxDB = InfluxDBFactory.connect(host)
	}

	def Series yearOHLC(int minutes) {
		return getOHLC(Duration.ofDays(365).toMinutes() as int, minutes) // 01-01-2017
	}

	def getOHLC(int candlesticks, int minutes) {
		val query = '''SELECT first(price) as open, max(price) as high, min(price) as low, last(price) as close, sum(amount) as volume FROM tradesBTCUSD.autogen.trade WHERE time > now() - «candlesticks*minutes»m GROUP BY time(«minutes»m)'''
		println("Running query: "+query)
		query(query).results.get(0).series.get(0)
	}

	def QueryResult query(Object... query) {
		influxDB.query(new Query(query.join(" "), tradesDB))
	}
	
	def saveLatestDate(int minutes) {
		val csv = Data.asCSV(Data.asDataset(yearOHLC(minutes)))
		Files.write(new File("data/ohlc-2017.csv").toPath, csv.split("\n"))
	}

	def static void main(String[] args) {
		extension val it = new Database(args.get(0))
		saveLatestDate(1)
	}

}
