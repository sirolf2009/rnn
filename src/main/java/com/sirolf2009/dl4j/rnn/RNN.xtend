package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.Data.DataFormat
import com.sirolf2009.dl4j.rnn.Data.PrepareData
import com.sirolf2009.dl4j.rnn.indicator.RnnCloseIndicator
import com.sirolf2009.progressbar.ProgressBar
import com.sirolf2009.progressbar.Styles
import java.io.File
import java.text.SimpleDateFormat
import java.time.Duration
import java.util.Date
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.eval.RegressionEvaluation
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
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.lossfunctions.LossFunctions

import static com.sirolf2009.dl4j.rnn.Data.*

import static extension com.sirolf2009.dl4j.rnn.ChartUtil.*

@org.eclipse.xtend.lib.annotations.Data
class RNN {

	static val rawDataFile = "ohlc-2017.csv"

	val int numberOfTimesteps
	val int miniBatchSize
	val int epochs

	def train() {
		extension val dataformat = new ProgressBar.Builder().name("Preparing "+rawDataFile).action(new PrepareData(rawDataFile, numberOfTimesteps, miniBatchSize)).style(Styles.ASCII).terminalWidth(250).build().get()

		println("Reading data with " + numOfVariables + " columns")
		extension val datasets = Data.getData(dataformat)

		println("Configuring the net")
		val builder = new NeuralNetConfiguration.Builder() => [
			optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
			iterations(1)
			weightInit = WeightInit.XAVIER
			updater = Updater.NESTEROVS
			learningRate = 0.1
			l2(0.001)
			useRegularization = true
		]
		val config = builder.list() => [
			layer(0, new GravesLSTM.Builder().rmsDecay(0.95).activation(Activation.TANH).updater(Updater.RMSPROP).nIn(numOfVariables).nOut(numOfVariables * 3).build())
			layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).momentum(0.9).activation(Activation.IDENTITY).nIn(numOfVariables * 3).nOut(numOfVariables).build())
		]
		val net = new MultiLayerNetwork(config.build())
		net.init()

		println("Training")
//		return new ProgressBar.Builder().name("Training").action(new Training.VisualTraining(net, rawDataFile, datasets, normalizer, dataformat, epochs)).style(Styles.ASCII).terminalWidth(250).build().get()
		return new ProgressBar.Builder().name("Training").action(new Training.EarlyStoppingTraining(net, datasets, epochs, numberOfTimesteps)).style(Styles.ASCII).terminalWidth(250).build().get()
	}

	def predict(MultiLayerNetwork net, int trainSize) {
		val db = new Database("http://database:8086")
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
		net.save(locationToSave)
	}

	def static save(MultiLayerNetwork net, File locationToSave) {
		val saveUpdater = true
		ModelSerializer.writeModel(net, locationToSave, saveUpdater)
	}

	def static load(String path) {
		println("Loading from " + path)
		val net = ModelSerializer.restoreMultiLayerNetwork(path)
		println("Network loaded")
		return net
	}

	def static void main(String[] args) {
		val forward = 60
		val batch = 1
		val epochs = 10
		new RNN(forward, batch, epochs) => [
//			new Database("http://198.211.120.29:8086").saveLatestDate(1)
			val net = train()
			net.save()
//			val net = "networks/early-stopping-2017-07-27T20-08-30/bestModel.bin".load()
		]
	}

}
