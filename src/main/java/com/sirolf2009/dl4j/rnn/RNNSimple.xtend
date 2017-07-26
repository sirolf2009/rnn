package com.sirolf2009.dl4j.rnn

import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.time.Duration
import java.util.Date
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

@org.eclipse.xtend.lib.annotations.Data
class RNNSimple {

	static val baseDir = new File("src/main/resources")
	static val featuresDirTrain = new File(baseDir, "features_train")
	static val labelsDirTrain = new File(baseDir, "labels_train")
	static val featuresDirTest = new File(baseDir, "features_test")
	static val labelsDirTest = new File(baseDir, "labels_test")
	static val rawDataFile = "ohlc-2017.csv"

	val int trainSize
	val int testSize
	val int numberOfTimesteps
	val int miniBatchSize
	val int epochs

	def init() {
		val trainingAndTestingSet = prepareTrainAndTest(trainSize, testSize, numberOfTimesteps)
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
//			seed = 140
			optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
			iterations(1)
			weightInit = WeightInit.XAVIER
			updater = Updater.NESTEROVS
			momentum = 0.5
			learningRate = 0.01
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

		val collection = new XYSeriesCollection()
		val trainArray = createIndArrayFromStringList(rawStrings, numOfVariables, 0, trainSize)
		val testArray = createIndArrayFromStringList(rawStrings, numOfVariables, trainSize, testSize)
		collection.createSeries(trainArray, 0, "Train data")
		collection.createSeries(testArray, trainSize - 1, "Actual test data")
		collection.plotDataset("Training")

		(0 ..< epochs).forEach [
			net.fit(trainIter)
			trainIter.reset()

			println('''Epoch «it» complete''')
			while(trainIter.hasNext()) {
				val data = trainIter.next()
				net.rnnTimeStep(data.featureMatrix)
			}
			trainIter.reset()

			val data = testIter.next()
			testIter.reset()
			val predicted = net.rnnTimeStep(data.featureMatrix)
			normalizer.revertLabels(predicted)

			collection.createSeries(predicted, trainSize - 1, "Epoch: " + it)
			net.rnnClearPreviousState()
		]

		net.predict()

		return net
	}

	def predict(MultiLayerNetwork net) {
		val db = new Database("http://198.211.120.29:8086")
		val dataset = Data.asDataset(db.getOHLC(trainSize, 1))
		val file = new File("src/main/resources/temp_0.csv")
		Files.write(file.toPath, Data.asCSV(dataset).split("\n"))
		val predictFeatures = new CSVSequenceRecordReader()
		predictFeatures.initialize(new NumberedFileInputSplit("src/main/resources/temp_%d.csv", 0, 0))
		val predictLabels = new CSVSequenceRecordReader()
		predictLabels.initialize(new NumberedFileInputSplit("src/main/resources/temp_%d.csv", 0, 0))
		val predictIter = new SequenceRecordReaderDataSetIterator(predictFeatures, predictLabels, miniBatchSize, -1, true)

		val normalizerP = new NormalizerMinMaxScaler(0, 1)
		normalizerP.fitLabel(true)
		normalizerP.fit(predictIter)
		predictIter.preProcessor = normalizerP

		val predictData = predictIter.next.featureMatrix
		val predicted = net.rnnTimeStep(predictData)
		val rows = predicted.shape.get(2)
		(0 ..< rows).forEach [ it, index |
			println("Close Predicted: " + predicted.slice(0).slice(0).getDouble(it))
		]
		normalizerP.revertLabels(predicted)

		dataset.get(0).forEach [
			println(time + " Close Actual   : " + value)
		]
		(0 ..< rows).forEach [ it, index |
			println(new Date(dataset.get(0).last.time.time + Duration.ofMinutes(index).toMillis) + " Close Predicted: " + predicted.slice(0).slice(0).getDouble(it))
		]
		(0 ..< Math.min(50, rows)).forEach [ it, index |
			val time = dataset.get(0).last.time.time + Duration.ofMinutes(index).toMillis
			val price = predicted.slice(0).slice(0).getDouble(it)
			println('''plotshape(time == «time» ? «price» : na, style=shape.circle, location=location.absolute, color=aqua, offset=«Duration.ofHours(2).toMinutes()»)''')
		]
		val collection = new XYSeriesCollection()
		val RecentArray = createIndArrayFromStringList(Data.asCSV(dataset).split("\n"), 5, 0, dataset.get(0).size())
		collection.createSeries(RecentArray, 0, "Recent data")
		collection.createSeries(predicted, RecentArray.shape.get(2), 0, "Predicted data Close")
		collection.createSeries(predicted, RecentArray.shape.get(2) - 1, 1, "Predicted data High")
		collection.createSeries(predicted, RecentArray.shape.get(2) - 1, 2, "Predicted data Low")
		collection.createSeries(predicted, RecentArray.shape.get(2) - 1, 3, "Predicted data Open")
		collection.plotDataset("Prediction")
	}

	def static void main(String[] args) {
//		val train = 4000
//		val test = 50
//		val forward = 50
		val train = 1000
		val test = 30
		val forward = test
		val batch = 10
		val epochs = 50
		new RNNSimple(train, test, forward, batch, epochs) => [
			init()
		]
	}

	def static plotDataset(XYSeriesCollection collection, String titleChart) {
		val title = rawDataFile
		val xAxisLabel = "Timestep"
		val yAxisLabel = "Close in $"
		val orientation = PlotOrientation.VERTICAL
		val legend = true
		val tooltips = false
		val urls = false
		val chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, collection, orientation, legend, tooltips, urls);
		val plot = chart.XYPlot
		val rangeAxis = plot.getRangeAxis() as NumberAxis
		rangeAxis.autoRange = true
		rangeAxis.autoRangeIncludesZero = false
		val panel = new ChartPanel(chart)
		val frame = new JFrame()
		frame.add(panel)
		frame.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
		frame.pack()
		frame.title = titleChart
		RefineryUtilities.centerFrameOnScreen(frame)
		frame.visible = true
		frame.setExtendedState(frame.getExtendedState().bitwiseOr(JFrame.MAXIMIZED_BOTH))
	}

	def static createSeries(XYSeriesCollection collection, INDArray data, int offset, String name) {
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
		val path = new File("src/main/resources/" + rawDataFile).toPath()
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
