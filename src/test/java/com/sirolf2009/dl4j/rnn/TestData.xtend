package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.model.Point
import com.sirolf2009.dl4j.rnn.model.TimeSeries
import java.util.Date
import org.junit.Test
import com.sirolf2009.dl4j.rnn.model.Dataset

class TestData {

	@Test
	def void testStrings() {
		val rawString = #[
			"1,2,3,4",
			"5,6,7,8"
		]
		val numOfVariables = 4
		val start = 0
		val length = rawString.size()

		val array = Data.createIndArrayFromStringList(rawString, numOfVariables, start, length)
		println(array)

		println(array.slice(1))
	}
	
	@Test
	def void testDataset() {
		val series1 = new TimeSeries("series 1", #[
			new Point(1d, new Date(0)),
			new Point(2d, new Date(1)),
			new Point(3d, new Date(2)),
			new Point(4d, new Date(3))
		])
		val series2 = new TimeSeries("series 2", #[
			new Point(10d, new Date(0)),
			new Point(20d, new Date(1)),
			new Point(30d, new Date(2)),
			new Point(40d, new Date(3))
		])
		val dataset = new Dataset()
		dataset += #[series1, series2]
		
		println(Data.createIndArrayFromDataset(dataset, 3))
	}

}
