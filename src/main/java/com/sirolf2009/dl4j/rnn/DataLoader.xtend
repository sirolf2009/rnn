package com.sirolf2009.dl4j.rnn

import eu.verdelhan.ta4j.Tick
import eu.verdelhan.ta4j.TimeSeries
import java.io.File
import java.nio.file.Files
import java.time.Duration
import java.time.Instant
import java.time.ZoneId
import java.time.ZonedDateTime
import java.util.List

import static extension java.lang.Double.parseDouble
import com.sirolf2009.progressbar.ProgressBar
import com.sirolf2009.progressbar.Styles

class DataLoader {
	
	def static TimeSeries loadOHLCV2017() {
		loadOHLCV("data/ohlc-2017.csv", "ohlc-2017");
	}
	
	def static loadOHLCV(String file, String name) {
		new File(file).loadOHLCV(name)
	}
	
	def static loadOHLCV(File file, String name) {
		Files.readAllLines(file.toPath()).loadOHLCV(name)
	}
	
	def static loadOHLCV(List<String> lines, String name) {
		return new TimeSeries(name, lines.map[
			val date = ZonedDateTime.ofInstant(Instant.ofEpochMilli(lines.indexOf(it)), ZoneId.systemDefault) 
			val cols = split(",")
			val close = cols.get(0).parseDouble()
			val high = cols.get(1).parseDouble()
			val low = cols.get(2).parseDouble()
			val open = cols.get(3).parseDouble()
			val volume = cols.get(4).parseDouble()
			return new Tick(date, open, high, low, close, volume) 
		])
	}
	
	def static TimeSeries loadBitstampSeries(Duration interval) {
		return new ProgressBar.Builder().name("Loading Bitstamp Data").action(new CsvTradesLoader(interval)).terminalWidth(250).style(Styles.ASCII).build().get()
	}
	
}