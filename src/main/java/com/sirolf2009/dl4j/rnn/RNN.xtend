package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.indicator.RnnCloseIndicator
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.text.SimpleDateFormat
import java.time.Duration
import java.util.Date
import java.util.List
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.jfree.data.xy.XYSeriesCollection
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.lossfunctions.LossFunctions

import static com.sirolf2009.dl4j.rnn.Data.*

import static extension com.sirolf2009.dl4j.rnn.ChartUtil.*
import static extension java.nio.file.Files.*
import static extension org.apache.commons.io.FileUtils.*

@org.eclipse.xtend.lib.annotations.Data
class RNN {

	static val baseDir = new File("data")
	static val featuresDirTrain = new File(baseDir, "features_train")
	static val labelsDirTrain = new File(baseDir, "labels_train")
	static val featuresDirTest = new File(baseDir, "features_test")
	static val labelsDirTest = new File(baseDir, "labels_test")
	static val rawDataFile = "ohlc-2017.csv"

	val int numberOfTimesteps
	val int miniBatchSize
	val int epochs
	val boolean ui

	def train() {
		println("Preparing data")
		extension val dataformat = prepareTrainAndTest(numberOfTimesteps, miniBatchSize)

		println("Reading data")
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

		val normalizer = new NormalizerMinMaxScaler(-1, 1)
		normalizer.fitLabel(true)
		normalizer.fit(trainIter)
		trainIter.reset()
		trainIter.preProcessor = normalizer
		testIter.preProcessor = normalizer

		println("Configuring the net")
		val builder = new NeuralNetConfiguration.Builder() => [
			optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
			iterations(1)
			weightInit = WeightInit.XAVIER
			updater = Updater.NESTEROVS
			momentum = 0.5
			learningRate = 0.1
		]
		val config = builder.list() => [
			layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numOfVariables).nOut(numOfVariables * 5).build())
			layer(1, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numOfVariables * 5).nOut(numOfVariables * 5).build())
			layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(numOfVariables * 5).nOut(numOfVariables).build())
		]
		val net = new MultiLayerNetwork(config.build())
		net.init()

		val networkFolder = new File("networks", "early-stopping-" + new SimpleDateFormat("yyyy-MM-dd'T'HH-mm-ss").format(new Date()))
		networkFolder.mkdirs()
		val earlyStopping = new EarlyStoppingConfiguration.Builder().epochTerminationConditions(new MaxEpochsTerminationCondition(epochs)).scoreCalculator(new ScoreCalculatorBitstamp(numberOfTimesteps)).modelSaver(new LocalFileModelSaver(networkFolder.absolutePath)).build()
		val earlyStoppingTrainer = new EarlyStoppingTrainer(earlyStopping, net, trainIter)
		val result = earlyStoppingTrainer.fit()

		println('''Termination Reason : «result.terminationReason»''')
		println('''Termination Details: «result.terminationDetails»''')
		println('''Epochs          : «result.totalEpochs»''')
		println('''Best Epoch      : «result.bestModelEpoch»''')
		println('''Best Epoch Score: «result.bestModelScore»''')

		return result.bestModel
//		val collection = new XYSeriesCollection()
//		collection.createSeries(trainingArray, 0, "Train data")
//		collection.createSeries(testingArray, trainSize - 1, "Actual test data")
//		collection.plotDataset("Training", rawDataFile)
//
//		if(ui) {
//			UIServer.instance => [
//				val storage = new InMemoryStatsStorage()
//				attach(storage)
//				net.listeners = new StatsListener(storage)
//			]
//		}
//
//		val start = System.currentTimeMillis()
//		(0 ..< epochs).forEach [
//			net.fit(trainIter)
//			trainIter.reset()
//
//			println('''Epoch «it» complete''')
//			while(trainIter.hasNext()) {
//				val data = trainIter.next()
//				net.rnnTimeStep(data.featureMatrix)
//			}
//			trainIter.reset()
//
//			val data = testIter.next()
//			testIter.reset()
//			val predicted = net.rnnTimeStep(data.featureMatrix)
//			normalizer.revertLabels(predicted)
//
//			collection.createSeries(predicted, trainSize - 1, "Epoch: " + it)
//			net.rnnClearPreviousState()
//		]
//		println("Training completed in " + Duration.ofMillis(System.currentTimeMillis() - start))
//
//		net.predict(trainSize)
//		net.save()
//
//		return net
	}

	def predict(MultiLayerNetwork net, int trainSize) {
		val db = new Database("http://198.211.120.29:8086")
		val dataset = Data.asDataset(db.getOHLC(trainSize, 1))

		val predictData = Data.createIndArrayFromDataset(dataset, numberOfTimesteps)
		val normalizerP = new NormalizerMinMaxScaler(-1, 1)
		normalizerP.fitLabel(true)
		normalizerP.fit(new DataSet(predictData, predictData))

		normalizerP.transform(predictData)
		val predicted = net.rnnTimeStep(predictData)
		normalizerP.revertLabels(predicted)

		val collection = new XYSeriesCollection()
		val RecentArray = createIndArrayFromStringList(Data.asCSV(dataset).split("\n"), 5, 0, dataset.get(0).size())
		collection.createSeries(RecentArray, 0, "Recent data")
		collection.createSeries(predicted, RecentArray.shape.get(2), 0, "Predicted data Close")
		collection.createSeries(predicted, RecentArray.shape.get(2) - 1, 1, "Predicted data High")
		collection.createSeries(predicted, RecentArray.shape.get(2) - 1, 2, "Predicted data Low")
		collection.createSeries(predicted, RecentArray.shape.get(2) - 1, 3, "Predicted data Open")
		collection.plotDataset("Prediction", rawDataFile)
	}

	def static save(MultiLayerNetwork net) {
		val locationToSave = new File("networks/" + new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss").format(new Date()) + ".zip")
		val saveUpdater = true
		ModelSerializer.writeModel(net, locationToSave, saveUpdater)
	}

	def static load(String path) {
		println("Loading from "+path)
		val net = ModelSerializer.restoreMultiLayerNetwork(path)
		println("Network loaded")
		return net
	}

	def static void main(String[] args) {
		val forward = 60
		val batch = 10
		val epochs = 50
		val ui = false
		new RNN(forward, batch, epochs, ui) => [
//			new Database("http://198.211.120.29:8086").saveLatestDate(1)
			val net = train()
//			val net = "networks/early-stopping-2017-07-27T12-27-26/bestModel.bin".load()
			println("Loading timeseries")
			val series = DataLoader.loadBitstampSeries(Duration.ofMinutes(1))
//			val series = DataLoader.loadOHLCV2017
			println("Creating indicator")
			val indicator = new RnnCloseIndicator(series, net, forward)
			println("Backtesting long")
			val backTestLong = ScoreCalculatorBitstamp.backtestLong(net, indicator, forward)
			println("Backtesting short")
			val backTestShort = ScoreCalculatorBitstamp.backtestShort(net, indicator, forward)
			val profitLong = backTestLong.map [
				val profitForTrade = (exit.price.toDouble-entry.price.toDouble)
				val fees = (entry.price.toDouble*0.002+exit.price.toDouble*0.002)
				println("buy at " + entry.price.toDouble + " exit at " + exit.price.toDouble + " Profit: " + profitForTrade+" Fees: "+fees)
				profitForTrade - fees
			].reduce[a,b|a+b]
			val profitShort = backTestShort.map [
				val profitForTrade = (entry.price.toDouble-exit.price.toDouble)
				val fees = (entry.price.toDouble*0.002+exit.price.toDouble*0.002)
				println("sell at " + entry.price.toDouble + " exit at " + exit.price.toDouble + " Profit: " + profitForTrade+" Fees: "+fees)
				profitForTrade - fees
			].reduce[a,b|a+b]
			println("Profit Long : " + profitLong)
			println("Profit Short: " + profitShort)
			ChartUtil.plotDataset(ChartUtil.createOHLCSeries(series, "Bitstamp"), ChartUtil.createSeries(indicator, "rnn"), "rnn", "rnn")
		]
	}

	def static DataFormat prepareTrainAndTest(int numberOfTimesteps, int miniBatchSize) {
		val start = System.currentTimeMillis()
		val path = new File("data/" + rawDataFile).toPath()
		val rawStrings = path.readAllLines
		println(Duration.ofMillis(System.currentTimeMillis - start) + " read raw data")
		val numOfVariables = rawStrings.numOfVariables

		featuresDirTrain.clean()
		labelsDirTrain.clean()
		featuresDirTest.clean()
		labelsDirTest.clean()

		val trainingLines = rawStrings.size() - numberOfTimesteps - numberOfTimesteps - numberOfTimesteps
		val trainSize = trainingLines - (trainingLines % miniBatchSize)

		(0 .. trainSize).toList.parallelStream.forEach [
			val featuresPath = Paths.get('''«featuresDirTrain.absolutePath»/train_«it».csv''')
			val labelsPath = Paths.get('''«labelsDirTrain.absolutePath»/train_«it».csv''')
			(0 .. numberOfTimesteps).forEach [ step |
				Files.write(featuresPath, rawStrings.get(it + step).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
			]
			Files.write(labelsPath, rawStrings.get(it + numberOfTimesteps).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
		]
		println(Duration.ofMillis(System.currentTimeMillis - start) + " created training data")

		(trainSize .. (numberOfTimesteps + trainSize)).toList.parallelStream.forEach [
			val featuresPath = Paths.get('''«featuresDirTest.absolutePath»/test_«it».csv''')
			val labelsPath = Paths.get('''«labelsDirTest.absolutePath»/test_«it».csv''')
			(0 .. numberOfTimesteps).forEach [ step |
				Files.write(featuresPath, rawStrings.get(it + step).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
			]
			Files.write(labelsPath, rawStrings.get(it + numberOfTimesteps).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
		]
		println(Duration.ofMillis(System.currentTimeMillis - start) + " created testing data")

		val trainingArray = createIndArrayFromStringList(rawStrings, numOfVariables, 0, trainSize)
		println(Duration.ofMillis(System.currentTimeMillis - start) + " created training array")
		val testingArray = createIndArrayFromStringList(rawStrings, numOfVariables, trainSize, numberOfTimesteps)
		println(Duration.ofMillis(System.currentTimeMillis - start) + " created testing array")

		return new DataFormat(trainSize, numberOfTimesteps, numberOfTimesteps, numOfVariables, trainingArray, testingArray)
	}

	@org.eclipse.xtend.lib.annotations.Data
	static class DataFormat {

		val int trainSize
		val int testSize
		val int numberOfTimesteps
		val int numOfVariables
		val INDArray trainingArray
		val INDArray testingArray

	}

	def static clean(File folder) {
		folder.mkdirs()
		folder.cleanDirectory()
	}

	def static getNumOfVariables(List<String> rawStrings) {
		return rawStrings.get(0).split(",").length
	}

}
