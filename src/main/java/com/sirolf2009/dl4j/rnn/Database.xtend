package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.model.Dataset
import com.sirolf2009.dl4j.rnn.model.Point
import com.sirolf2009.dl4j.rnn.model.TimeSeries
import java.io.File
import java.nio.file.Files
import java.text.SimpleDateFormat
import java.time.Duration
import java.util.stream.Collectors
import java.util.stream.Stream
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

	def asDataset(Series series) {
		val dataset = new Dataset()
		dataset += series.values.parallelStream.filter[!(get(0) as String).isEmpty].filter[get(1) !== null].flatMap [
			val sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'")
			val date = sdf.parse(get(0) as String)
			val open = "open" -> new Point(get(1) as Double, date)
			val high = "high" -> new Point(get(2) as Double, date)
			val low = "low" -> new Point(get(3) as Double, date)
			val close = "close" -> new Point(get(4) as Double, date)
			val vol = "vol" -> new Point(get(5) as Double, date)
			return Stream.of(open, high, low, close, vol)
		].filter[value !== null].collect(Collectors.groupingBy([key], Collectors.mapping([value], Collectors.toList))).entrySet.map [
			new TimeSeries(key, value.stream.sorted[a, b|a.time.compareTo(b.time)].collect(Collectors.toList))
		].sortWith[a,b|a.name.compareTo(b.name)]
		dataset
	}

	def asCSV(Dataset dataset) {
		dataset.stream.flatMap[stream.map[point|name -> point]].collect(Collectors.groupingBy([value.getTime()])).entrySet.stream.sorted[a, b|a.key.compareTo(b.key)].map [
			value.stream.sorted[a, b|a.key.compareTo(b.key)].map[value.value + ""].reduce[a, b|a + ", " + b].orElse("")
		].reduce[a, b|a + "\n" + b].orElse("")
	}

	def QueryResult query(Object... query) {
		influxDB.query(new Query(query.join(" "), tradesDB))
	}
	
	def saveLatestDate(int minutes) {
		val csv = yearOHLC(minutes).asDataset().asCSV()
		Files.write(new File("src/main/resources/ohlc-2017.csv").toPath, csv.split("\n"))
	}

	def static void main(String[] args) {
		extension val it = new Database(args.get(0))
		saveLatestDate(1)
	}

}
