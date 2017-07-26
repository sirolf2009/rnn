package com.sirolf2009.dl4j.rnn

import eu.verdelhan.ta4j.Decimal
import eu.verdelhan.ta4j.Indicator
import java.util.Date
import javax.swing.JFrame
import javax.swing.WindowConstants
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.time.Minute
import org.jfree.data.time.TimeSeries
import org.jfree.data.time.TimeSeriesCollection
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import org.jfree.ui.RefineryUtilities
import org.nd4j.linalg.api.ndarray.INDArray
import org.jfree.data.xy.DefaultHighLowDataset
import org.jfree.data.xy.OHLCDataset
import java.util.List
import java.util.Optional
import org.jfree.chart.renderer.xy.CandlestickRenderer
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer
import java.awt.Color

class ChartUtil {
	
	def static plotDataset(XYSeriesCollection collection, String titleFrame, String titleChart) {
		val xAxisLabel = "Timestep"
		val yAxisLabel = "Close in $"
		val orientation = PlotOrientation.VERTICAL
		val legend = true
		val tooltips = false
		val urls = false
		val chart = ChartFactory.createXYLineChart(titleChart, xAxisLabel, yAxisLabel, collection, orientation, legend, tooltips, urls)
		val plot = chart.XYPlot
		val rangeAxis = plot.getRangeAxis() as NumberAxis
		rangeAxis.autoRange = true
		rangeAxis.autoRangeIncludesZero = false
		val panel = new ChartPanel(chart)
		val frame = new JFrame()
		frame.add(panel)
		frame.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
		frame.pack()
		frame.title = titleFrame
		RefineryUtilities.centerFrameOnScreen(frame)
		frame.visible = true
		frame.setExtendedState(frame.getExtendedState().bitwiseOr(JFrame.MAXIMIZED_BOTH))
	}
	
	def static plotIndicator(Indicator<Decimal> indicator, String name) {
		val collection = new TimeSeriesCollection()
		collection.addSeries(indicator.createSeries(name))
		plotDataset(collection, name, name)
	}
	
	def static plotDataset(TimeSeriesCollection collection, String titleFrame, String titleChart) {
		val xAxisLabel = "Timestep"
		val yAxisLabel = "Close in $"
		val legend = true
		val tooltips = false
		val urls = false
		val chart = ChartFactory.createTimeSeriesChart(titleChart, xAxisLabel, yAxisLabel, collection, legend, tooltips, urls)
		val plot = chart.XYPlot
		val rangeAxis = plot.getRangeAxis() as NumberAxis
		rangeAxis.autoRange = true
		rangeAxis.autoRangeIncludesZero = false
		val panel = new ChartPanel(chart)
		val frame = new JFrame()
		frame.add(panel)
		frame.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
		frame.pack()
		frame.title = titleFrame
		RefineryUtilities.centerFrameOnScreen(frame)
		frame.visible = true
		frame.setExtendedState(frame.getExtendedState().bitwiseOr(JFrame.MAXIMIZED_BOTH))
	}
	
	def static plotDataset(OHLCDataset dataset, String titleFrame, String titleChart) {
		plotDataset(dataset, Optional.empty(), titleFrame, titleChart);
	}
	
	def static plotDataset(OHLCDataset dataset, TimeSeries indicator, String titleFrame, String titleChart) {
		val collection = new TimeSeriesCollection()
		collection.addSeries(indicator)
		plotDataset(dataset, Optional.of(#[collection]), titleFrame, titleChart)
	}
	
	def static plotDataset(OHLCDataset dataset, TimeSeriesCollection indicators, String titleFrame, String titleChart) {
		plotDataset(dataset, Optional.of(#[indicators]), titleFrame, titleChart)
	}
	
	def static plotDataset(OHLCDataset dataset, List<TimeSeriesCollection> indicators, String titleFrame, String titleChart) {
		plotDataset(dataset, Optional.of(indicators), titleFrame, titleChart)
	}
	
	def private static plotDataset(OHLCDataset dataset, Optional<List<TimeSeriesCollection>> indicators, String titleFrame, String titleChart) {
		val xAxisLabel = "Timestep"
		val yAxisLabel = "Close in $"
		val legend = true
		val chart = ChartFactory.createCandlestickChart(titleChart, xAxisLabel, yAxisLabel, dataset, legend)
		val renderer = new CandlestickRenderer()
		renderer.autoWidthMethod = CandlestickRenderer.WIDTHMETHOD_SMALLEST
		val plot = chart.XYPlot
		plot.renderer = renderer
		
		indicators.ifPresent[
			forEach[it, index|
				plot.setDataset(index+1, it)
				plot.mapDatasetToRangeAxis(index+1, 0)
				val rendererIndicator = new XYLineAndShapeRenderer(true, false)
				rendererIndicator.setSeriesPaint(index+1, Color.CYAN)
				plot.setRenderer(index+1, rendererIndicator)
			]
		]
		val rangeAxis = plot.getRangeAxis() as NumberAxis
		rangeAxis.autoRange = true
		rangeAxis.autoRangeIncludesZero = false
		val panel = new ChartPanel(chart)
		val frame = new JFrame()
		frame.add(panel)
		frame.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
		frame.pack()
		frame.title = titleFrame
		RefineryUtilities.centerFrameOnScreen(frame)
		frame.visible = true
		frame.setExtendedState(frame.getExtendedState().bitwiseOr(JFrame.MAXIMIZED_BOTH))
	}

	def public static createSeries(XYSeriesCollection collection, INDArray data, int offset, String name) {
		val rows = data.shape.get(2)
		val predicted = name.startsWith("Epoch")
		val series = new XYSeries(name)
		(0 ..< rows).forEach [
			if(predicted) {
				series.add(it + offset, data.slice(0).slice(0).getDouble(it))
			} else {
				series.add(it + offset, data.slice(0).getDouble(it))
			}
		]
		collection.addSeries(series)
	}

	def static createSeries(XYSeriesCollection collection, INDArray data, int offset, int index, String name) {
		val rows = data.shape.get(2)
		val series = new XYSeries(name)
		(0 ..< rows).forEach [
			series.add(it + offset, data.slice(0).slice(index).getDouble(it))
		]
		collection.addSeries(series)
	}
	
	def static createSeries(Indicator<Decimal> indicator, String name) {
        val chartTimeSeries = new TimeSeries(name)
        for (var int i = 0; i < indicator.timeSeries.getTickCount(); i++) {
            val tick = indicator.timeSeries.getTick(i)
            chartTimeSeries.add(new Minute(Date.from(tick.getEndTime().toInstant())), indicator.getValue(i).toDouble())
        }
        return chartTimeSeries
    }
    
    def static createOHLCSeries(eu.verdelhan.ta4j.TimeSeries series, String name) {
    	val ticks = series.getTickCount()
    	val dates = newArrayList()
    	val opens = newArrayList()
    	val highs = newArrayList()
    	val lows = newArrayList()
    	val closes = newArrayList()
    	val volumes = newArrayList()
    	(0 ..< ticks).forEach[
    		val it = series.getTick(it)
    		dates += new Date(endTime.toEpochSecond*1000)
    		opens += openPrice.toDouble()
    		highs += maxPrice.toDouble()
    		lows += minPrice.toDouble()
    		closes += closePrice.toDouble()
    		volumes += amount.toDouble()
    	]
    	return new DefaultHighLowDataset(name, dates, highs, lows, opens, closes, volumes)
    }
	
}