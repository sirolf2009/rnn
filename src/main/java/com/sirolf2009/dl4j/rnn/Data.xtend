package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.model.Dataset
import java.util.List
import org.nd4j.linalg.factory.Nd4j
import java.util.stream.Collectors

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
	
	def static createIndArrayFromDataset(Dataset dataset, int forward) {
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
	
}