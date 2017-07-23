package com.sirolf2009.dl4j.rnn

import javax.swing.JFrame
import javax.swing.WindowConstants
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import org.jfree.ui.RefineryUtilities
import org.nd4j.linalg.api.ndarray.INDArray

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
	
}