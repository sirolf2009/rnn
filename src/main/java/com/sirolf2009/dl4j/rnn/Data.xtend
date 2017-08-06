package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.model.Dataset
import com.sirolf2009.dl4j.rnn.model.Point
import com.sirolf2009.dl4j.rnn.model.TimeSeries
import de.oehme.xtend.contrib.Cached
import java.text.SimpleDateFormat
import java.util.List
import java.util.stream.Collectors
import java.util.stream.Stream
import org.influxdb.dto.QueryResult.Series
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class Data {
	
	def static createIndArrayFromStringList(List<String> rawString, int numOfVariables, int start, int length) {
		val stringList = rawString.subList(start, start + length)
		val double[][] primitives = Matrix.new2DDoubleArrayOfSize(numOfVariables, stringList.size())
		stringList.forEach [ it, i |
			val vals = split(",")
			vals.forEach [ it, j |
				primitives.get(j).set(i, Double.valueOf(vals.get(j)))
			]
		]
		return Nd4j.create(#[1, length], primitives)
	}
	
	def static INDArray createIndArrayFromDataset(Dataset dataset, int forward) {
		val initializationInput = Nd4j.zeros(1, dataset.size(), forward)
		val matrix = dataset.asMatrix(forward)
		matrix.forEach[array,i|
			array.forEach[value, j|
				initializationInput.putScalar(#[0, j, i], value)
			]
		]
		return initializationInput
	}
	
	def static asMatrix(Dataset dataset, int forward) {
		return dataset.stream.flatMap[series|
			series.stream.map[series.name -> it]
		].collect(Collectors.groupingBy([value.time])).entrySet.stream.sorted[a,b|a.key.compareTo(b.key)].limit(forward).map[
			value.stream.sorted[a,b|a.key.compareTo(b.key)].map[value.value].collect(Collectors.toList())
		].collect(Collectors.toList())
	}
	
	def static asDataset(Series series) {
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

	def static asCSV(Dataset dataset) {
		dataset.stream.flatMap[stream.map[point|name -> point]].collect(Collectors.groupingBy([value.getTime()])).entrySet.stream.sorted[a, b|a.key.compareTo(b.key)].map [
			value.stream.sorted[a, b|a.key.compareTo(b.key)].map[value.value + ""].reduce[a, b|a + ", " + b].orElse("")
		].reduce[a, b|a + "\n" + b].orElse("")
	}
	
}