package com.sirolf2009.dl4j.rnn

import java.io.File
import java.nio.file.Files
import java.time.Instant
import java.time.ZoneId
import java.time.ZonedDateTime
import java.util.List
import eu.verdelhan.ta4j.Tick
import eu.verdelhan.ta4j.TimeSeries

import static extension java.lang.Double.parseDouble

class DataLoader {
	
	def static loadOHLCV2017() {
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
	
}