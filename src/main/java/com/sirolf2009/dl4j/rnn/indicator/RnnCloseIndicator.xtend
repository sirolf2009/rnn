package com.sirolf2009.dl4j.rnn.indicator

import com.sirolf2009.dl4j.rnn.Data
import com.sirolf2009.dl4j.rnn.model.Dataset
import com.sirolf2009.dl4j.rnn.model.Point
import com.sirolf2009.dl4j.rnn.model.TimeSeries
import eu.verdelhan.ta4j.Decimal
import eu.verdelhan.ta4j.indicators.CachedIndicator
import java.util.Date
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler

class RnnCloseIndicator extends CachedIndicator<Decimal> {

	val MultiLayerNetwork net
	val int forward
	val Dataset dataset

	new(eu.verdelhan.ta4j.TimeSeries series, MultiLayerNetwork net, int forward) {
		super(series)
		this.net = net
		this.forward = forward
		this.dataset = series.dataset
	}

	override protected calculate(int index) {
		if(index < forward) {
			return timeSeries.getTick(index).closePrice
		}
		val input = Data.createIndArrayFromDataset(dataset.subset(index-forward, index), forward)
		val normalizer = new NormalizerMinMaxScaler(-1, 1)
		normalizer.fitLabel(true)
		normalizer.fit(new DataSet(input, input))
		normalizer.transform(input)
		val predicted = net.output(input, true)
		normalizer.revertLabels(predicted)
		//shape 1,5,forward
//		val now = predicted.slice(0).slice(0).getDouble(0)
		val future =predicted.slice(0).slice(0).getDouble(forward-1)
//		val diff = future - now
//		return timeSeries.getTick(index).closePrice.plus(Decimal.valueOf(diff))
		return Decimal.valueOf(future)
	}
	
	def getDataset(eu.verdelhan.ta4j.TimeSeries series) {
		val open = newArrayList()
		val high = newArrayList()
		val low = newArrayList()
		val close = newArrayList()
		val vol = newArrayList()
		(0 ..< series.tickCount).forEach[
			val tick = series.getTick(it)
			val time = new Date(tick.beginTime.toInstant.toEpochMilli)
			open += new Point(tick.openPrice.toDouble, time)
			high += new Point(tick.maxPrice.toDouble, time)
			low += new Point(tick.minPrice.toDouble, time)
			close += new Point(tick.closePrice.toDouble, time)
			vol += new Point(tick.volume.toDouble, time)
		]
		val dataset = new Dataset()
		dataset += new TimeSeries("Open", open)
		dataset += new TimeSeries("High", high)
		dataset += new TimeSeries("Low", low)
		dataset += new TimeSeries("Close", close)
		dataset += new TimeSeries("Volume", vol)
		return dataset
	}

}
