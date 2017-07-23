package com.sirolf2009.dl4j.rnn

import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.util.List
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
import org.jfree.data.xy.XYSeriesCollection
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.lossfunctions.LossFunctions

import static com.sirolf2009.dl4j.rnn.Data.*

import static extension com.sirolf2009.dl4j.rnn.ChartUtil.*
import static extension java.nio.file.Files.*
import static extension org.apache.commons.io.FileUtils.*

@org.eclipse.xtend.lib.annotations.Data class RNN {
	
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
		collection.plotDataset("Training", rawDataFile)

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
		val dataset = db.asDataset(db.getOHLC(trainSize, 1))

		val predictData = Data.createIndArrayFromDataset(dataset, numberOfTimesteps)
		val normalizerP = new NormalizerMinMaxScaler(0, 1)
		normalizerP.fitLabel(true)
		normalizerP.fit(new DataSet(predictData, predictData))

		normalizerP.transform(predictData)
		val predicted = net.rnnTimeStep(predictData)
		val rows = predicted.shape.get(2)
		(0 ..< rows).forEach [ it, index |
			println("Close Predicted: " + predicted.slice(0).slice(0).getDouble(it))
		]
		normalizerP.revertLabels(predicted)

		val collection = new XYSeriesCollection()
		val RecentArray = createIndArrayFromStringList(db.asCSV(dataset).split("\n"), 5, 0, dataset.get(0).size())
		collection.createSeries(RecentArray, 0, "Recent data")
		collection.createSeries(predicted, RecentArray.shape.get(2), 0, "Predicted data Close")
		collection.createSeries(predicted, RecentArray.shape.get(2) - 1, 1, "Predicted data High")
		collection.createSeries(predicted, RecentArray.shape.get(2) - 1, 2, "Predicted data Low")
		collection.createSeries(predicted, RecentArray.shape.get(2) - 1, 3, "Predicted data Open")
		collection.plotDataset("Prediction", rawDataFile)
	}

	def static void main(String[] args) {
//		val train = 4000
//		val test = 50
//		val forward = 50
		val train = 1050
		val test = 30
		val forward = test
		val batch = 10
		val epochs = 5
		new RNNSimple(train, test, forward, batch, epochs) => [
			init()
		]
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