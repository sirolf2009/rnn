package com.sirolf2009.dl4j.rnn

import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.util.List
import javax.swing.JFrame
import javax.swing.WindowConstants
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.eclipse.xtend.lib.annotations.Data
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import org.jfree.ui.RefineryUtilities
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import static extension java.nio.file.Files.*
import static extension org.apache.commons.io.FileUtils.*

@Data
class RNNSimple {

	static val baseDir = new File("src/main/resources")
	static val featuresDirTrain = new File(baseDir, "features_train")
	static val labelsDirTrain = new File(baseDir, "labels_train")
	static val featuresDirTest = new File(baseDir, "features_test")
	static val labelsDirTest = new File(baseDir, "labels_test")

	val int trainSize
	val int testSize
	val int numberOfTimestesp
	val int miniBatchSize

	def init() {
		val trainingAndTestingSet = prepareTrainAndTest(trainSize, testSize, numberOfTimestesp)
		val rawStrings = trainingAndTestingSet.key
		val numOfVariables = trainingAndTestingSet.value

		val trainFeatures = new CSVSequenceRecordReader()
		trainFeatures.initialize(new NumberedFileInputSplit('''«featuresDirTrain.absolutePath»/train_%d.csv''', 0, trainSize - 1))
		val trainLabels = new CSVSequenceRecordReader()
		trainLabels.initialize(new NumberedFileInputSplit('''«labelsDirTrain.absolutePath»/train_%d.csv''', 0, trainSize - 1))
		val trainIter = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, -1, true, AlignmentMode.ALIGN_END)

		val testFeatures = new CSVSequenceRecordReader()
		testFeatures.initialize(new NumberedFileInputSplit('''«featuresDirTest.absolutePath»/test_%d.csv''', trainSize, trainSize + testSize))
		val testLabels = new CSVSequenceRecordReader()
		testLabels.initialize(new NumberedFileInputSplit('''«labelsDirTest.absolutePath»/test_%d.csv''', trainSize, trainSize + testSize))
		val testIter = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, -1, true, AlignmentMode.ALIGN_END)

		val normalizer = new NormalizerMinMaxScaler(0, 1)
		normalizer.fitLabel(true)
		normalizer.fit(trainIter)
		trainIter.reset()
		trainIter.preProcessor = normalizer
		testIter.preProcessor = normalizer

		val builder = new NeuralNetConfiguration.Builder() => [
			seed = 140
			optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
			iterations(1)
			weightInit = WeightInit.XAVIER
			updater = Updater.NESTEROVS
			momentum = 0.1
			learningRate = 0.015
		]
		val config = builder.list() => [
			layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numOfVariables).nOut(10).build())
			layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(10).nOut(numOfVariables).build())
		]
		val net = new MultiLayerNetwork(config.build())
		net.init()

		net.listeners += #[
			new ScoreIterationListener(20)
		]

		val epochs = 50
		(0 ..< epochs).forEach [
			net.fit(trainIter)
			trainIter.reset()

//			println('''Epoch «it» complete. Time series evaluation:''')
//			val evaluation = new RegressionEvaluation(numOfVariables)
//			while(testIter.hasNext()) {
//				val data = testIter.next()
//				val features = data.featureMatrix
//				val labels = data.labels
//				val predicted = net.output(features, true)
//				evaluation.evalTimeSeries(labels, predicted)
//			}
//			println(evaluation.stats())
//			testIter.reset()
		]

		// plotting
		while(trainIter.hasNext()) {
			val data = trainIter.next()
			net.rnnTimeStep(data.featureMatrix)
		}
		trainIter.reset()

		val data = testIter.next()
		val predicted = net.rnnTimeStep(data.featureMatrix)
		normalizer.revertLabels(predicted)
		
		val trainArray = createIndArrayFromStringList(rawStrings, numOfVariables, 0, trainSize)
		val testArray = createIndArrayFromStringList(rawStrings, numOfVariables, trainSize, testSize)
		
		val collection = new XYSeriesCollection()
		collection.createSeries(trainArray, 0, "Train data")
		collection.createSeries(testArray, trainSize-1, "Actual test data")
		collection.createSeries(predicted, trainSize-1, "Predicted test data")
		collection.plotDataset()

		return net
	}

	def static void main(String[] args) {
		new RNNSimple(100, 20, 20, 10) => [
			init()
		]
	}
	
	def static plotDataset(XYSeriesCollection collection) {
		val title = "Regression example"
		val xAxisLabel = "Timestep"
		val yAxisLabel = "Number of passengers"
		val orientation = PlotOrientation.VERTICAL
		val legend = true
		val tooltips = false
		val urls = false
		val chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, collection, orientation, legend, tooltips, urls);
		val plot = chart.XYPlot
		val rangeAxis = plot.getRangeAxis() as NumberAxis
		rangeAxis.autoRange = true
		val panel = new ChartPanel(chart)
		val frame = new JFrame()
		frame.add(panel)
		frame.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
		frame.pack()
		frame.title = "Training Data"
		RefineryUtilities.centerFrameOnScreen(frame)
		frame.visible = true
	}
	
	def static createSeries(XYSeriesCollection collection, INDArray data, int offset, String name) {
		val rows = data.shape.get(2)
		val predicted = name.startsWith("Predicted")
		val series = new XYSeries(name)
		(0 ..< rows).forEach[
			if(predicted) {
				series.add(it+offset, data.slice(0).slice(0).getDouble(it))
			} else {
				series.add(it+offset, data.slice(0).getDouble(it))
			}
		]
		collection.addSeries(series)
	}

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

	def static Pair<List<String>, Integer> prepareTrainAndTest(int trainSize, int testSize, int numberOfTimesteps) {
		val path = new File("src/main/resources/orderbook2.csv").toPath()
		val rawStrings = path.readAllLines
		val numOfVariables = rawStrings.numOfVariables

		featuresDirTrain.clean()
		labelsDirTrain.clean()
		featuresDirTest.clean()
		labelsDirTest.clean()

		(0 .. trainSize).forEach [
			val featuresPath = Paths.get('''«featuresDirTrain.absolutePath»/train_«it».csv''')
			val labelsPath = Paths.get('''«labelsDirTrain.absolutePath»/train_«it».csv''')
			(0 .. numberOfTimesteps).forEach [ step |
				Files.write(featuresPath, rawStrings.get(it + step).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
			]
			Files.write(labelsPath, rawStrings.get(it + numberOfTimesteps).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
		]

		(trainSize .. (testSize + trainSize)).forEach [
			val featuresPath = Paths.get('''«featuresDirTest.absolutePath»/test_«it».csv''')
			val labelsPath = Paths.get('''«labelsDirTest.absolutePath»/test_«it».csv''')
			(0 .. numberOfTimesteps).forEach [ step |
				Files.write(featuresPath, rawStrings.get(it + step).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
			]
			Files.write(labelsPath, rawStrings.get(it + numberOfTimesteps).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
		]

		rawStrings -> numOfVariables
	}

	def static clean(File folder) {
		folder.mkdirs()
		folder.cleanDirectory()
	}

	def static getNumOfVariables(List<String> rawStrings) {
		return rawStrings.get(0).split(",").length
	}

}
