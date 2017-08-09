package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.Data.DataFormat
import com.sirolf2009.dl4j.rnn.Data.TrainAndTestData
import com.sirolf2009.progressbar.Action
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.EarlyStoppingResult
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.eval.RegressionEvaluation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.jfree.data.xy.XYSeriesCollection
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

import static extension com.sirolf2009.dl4j.rnn.ChartUtil.*

class Training {

	@org.eclipse.xtend.lib.annotations.Data static class VisualTraining extends Action<MultiLayerNetwork> {

		val MultiLayerNetwork net
		val String rawDataFile
		val extension TrainAndTestData trainAndTestData
		val extension DataFormat format
		val int epochs

		override call() throws Exception {
			val collection = new XYSeriesCollection()
			collection.createSeries(trainingArray, 0, 0, "Train Bid")
//			collection.createSeries(trainingArray, 0, 30, "Train Ask")
			collection.createSeries(testingArray, trainSize - 1, 0, "Bid")
//			collection.createSeries(testingArray, trainSize - 1, 30, "Ask")
			collection.plotDataset("Training", rawDataFile)

			(0 ..< epochs).forEach [
				net.fit(trainData)
				trainData.reset()
				progress()

				message = '''Epoch «it» complete'''
				while(trainData.hasNext()) {
					val data = trainData.next()
					net.rnnTimeStep(data.featureMatrix)
				}
				trainData.reset()

				val data = testData.next()
				testData.reset()
				val predicted = net.rnnTimeStep(data.featureMatrix)
				normalizer.revertLabels(predicted)

				collection.createSeries(predicted, trainSize - 1, 0, true, "Bid at epoch: " + it)
//				collection.createSeries(predicted, trainSize - 1, 31, true, "Ask at epoch: " + it)
				net.rnnClearPreviousState()

//				net.showRegressionEvaluation(testData, numOfVariables)
			]
			return net
		}

		override getWorkloadSize() {
			return epochs
		}

		def showRegressionEvaluation(MultiLayerNetwork net, DataSetIterator testDataIter, int numOfVariables) {
			val evaluation = new RegressionEvaluation(numOfVariables)
			while(testDataIter.hasNext()) {
				val t = testDataIter.next()
				val features = t.getFeatureMatrix()
				val labels = t.getLabels()
				val predicted = net.output(features, true)
				evaluation.evalTimeSeries(labels, predicted)
			}
			println(evaluation.stats())
			testDataIter.reset()
		}

	}

	@org.eclipse.xtend.lib.annotations.Data static class EarlyStoppingTraining extends Action<MultiLayerNetwork> {

		val MultiLayerNetwork net
		val extension TrainAndTestData trainAndTestData
		val int epochs
		val int numberOfTimesteps

		override call() throws Exception {
			val networkFolder = new File("networks", "early-stopping-" + new SimpleDateFormat("yyyy-MM-dd'T'HH-mm-ss").format(new Date()))
			networkFolder.mkdirs()
			val earlyStopping = new EarlyStoppingConfiguration.Builder().epochTerminationConditions(new MaxEpochsTerminationCondition(epochs)).evaluateEveryNEpochs(1).scoreCalculator(new ScoreCalculatorBitstamp(numberOfTimesteps)).build()
			val earlyStoppingTrainer = new EarlyStoppingTrainer(earlyStopping, net, trainData)
			earlyStoppingTrainer.listener = new EarlyStoppingListener<MultiLayerNetwork>() {
				override onCompletion(EarlyStoppingResult<MultiLayerNetwork> MultiLayerNetwork) {
					message = "Done"
				}
				override onEpoch(int epoch, double score, EarlyStoppingConfiguration<MultiLayerNetwork> arg2, MultiLayerNetwork arg3) {
					progress()
					message = "Epoch="+epoch+" Score="+score
				}
				override onStart(EarlyStoppingConfiguration<MultiLayerNetwork> arg0, MultiLayerNetwork arg1) {
					message = "Training"
				}
			}
			val result = earlyStoppingTrainer.fit()

			println('''Termination Reason : «result.terminationReason»''')
			println('''Termination Details: «result.terminationDetails»''')
			println('''Epochs          : «result.totalEpochs»''')
			println('''Best Epoch      : «result.bestModelEpoch»''')
			println('''Best Epoch Score: «result.bestModelScore»''')
			result.bestModel
		}

		override getWorkloadSize() {
			return epochs
		}

	}

	def earlyStoppingTraining(MultiLayerNetwork net, DataSetIterator trainIter, int epochs, int numberOfTimesteps) {
		val networkFolder = new File("networks", "early-stopping-" + new SimpleDateFormat("yyyy-MM-dd'T'HH-mm-ss").format(new Date()))
		networkFolder.mkdirs()
		val earlyStopping = new EarlyStoppingConfiguration.Builder().epochTerminationConditions(new MaxEpochsTerminationCondition(epochs)).evaluateEveryNEpochs(1).scoreCalculator(new ScoreCalculatorBitstamp(numberOfTimesteps)).build()
		val earlyStoppingTrainer = new EarlyStoppingTrainer(earlyStopping, net, trainIter)
		val result = earlyStoppingTrainer.fit()

		println('''Termination Reason : «result.terminationReason»''')
		println('''Termination Details: «result.terminationDetails»''')
		println('''Epochs          : «result.totalEpochs»''')
		println('''Best Epoch      : «result.bestModelEpoch»''')
		println('''Best Epoch Score: «result.bestModelScore»''')
		result.bestModel
	}
}
