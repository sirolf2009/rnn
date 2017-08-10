package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.model.Dataset
import com.sirolf2009.dl4j.rnn.model.Point
import com.sirolf2009.dl4j.rnn.model.TimeSeries
import com.sirolf2009.progressbar.ActionTimed
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.text.SimpleDateFormat
import java.util.List
import java.util.stream.Collectors
import java.util.stream.Stream
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode
import org.influxdb.dto.QueryResult.Series
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.factory.Nd4j

import static com.sirolf2009.dl4j.rnn.Properties.*

import static extension org.apache.commons.io.FileUtils.*

class Data {

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

	def static INDArray createIndArrayFromDataset(Dataset dataset, int forward) {
		val initializationInput = Nd4j.zeros(1, dataset.size(), forward)
		val matrix = dataset.asMatrix(forward)
		matrix.forEach [ array, i |
			array.forEach [ value, j |
				initializationInput.putScalar(#[0, j, i], value)
			]
		]
		return initializationInput
	}

	def static asMatrix(Dataset dataset, int forward) {
		return dataset.stream.flatMap [ series |
			series.stream.map[series.name -> it]
		].collect(Collectors.groupingBy([value.time])).entrySet.stream.sorted[a, b|a.key.compareTo(b.key)].limit(forward).map [
			value.stream.sorted[a, b|a.key.compareTo(b.key)].map[value.value].collect(Collectors.toList())
		].collect(Collectors.toList())
	}

	def static asDataset(Series series) {
		val dataset = new Dataset()
		dataset += series.values.parallelStream.filter[!(get(0) as String).isEmpty].filter[get(1) !== null].flatMap [
			val sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'")
			val date = sdf.parse(get(0) as String)
			val open = "open" -> new Point(get(1) as Double, date)
			val high = "high" -> new Point(get(2) as Double, date)
			val low = "low" -> new Point(get(3) as Double, date)
			val close = "close" -> new Point(get(4) as Double, date)
			val vol = "vol" -> new Point(get(5) as Double, date)
			return Stream.of(open, high, low, close, vol)
		].filter[value !== null].collect(Collectors.groupingBy([key], Collectors.mapping([value], Collectors.toList))).entrySet.map [
			new TimeSeries(key, value.stream.sorted[a, b|a.time.compareTo(b.time)].collect(Collectors.toList))
		].sortWith[a, b|a.name.compareTo(b.name)]
		dataset
	}

	def static asCSV(Dataset dataset) {
		dataset.stream.flatMap[stream.map[point|name -> point]].collect(Collectors.groupingBy([value.getTime()])).entrySet.stream.sorted[a, b|a.key.compareTo(b.key)].map [
			value.stream.sorted[a, b|a.key.compareTo(b.key)].map[value.value + ""].reduce[a, b|a + ", " + b].orElse("")
		].reduce[a, b|a + "\n" + b].orElse("")
	}

	def static normalize(List<DataSetIterator> iters) {
		val normalizer = new NormalizerMinMaxScaler(-1, 1)
		normalizer.fitLabel(true)
		normalizer.fit(iters.get(0))
		iters.get(0).reset()
		iters.forEach[preProcessor = normalizer]
		return normalizer
	}

	def static getData(extension DataFormat format) {
		val trainFeatures = new CSVSequenceRecordReader()
		trainFeatures.initialize(new NumberedFileInputSplit('''«featuresDirTrain.absolutePath»/train_%d.csv''', 0, trainSize - 1))
		val trainLabels = new CSVSequenceRecordReader()
		trainLabels.initialize(new NumberedFileInputSplit('''«labelsDirTrain.absolutePath»/train_%d.csv''', 0, trainSize - 1))
		val trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, -1, true, AlignmentMode.ALIGN_END)

		val testFeatures = new CSVSequenceRecordReader()
		testFeatures.initialize(new NumberedFileInputSplit('''«featuresDirTest.absolutePath»/test_%d.csv''', trainSize, trainSize + testSize))
		val testLabels = new CSVSequenceRecordReader()
		testLabels.initialize(new NumberedFileInputSplit('''«labelsDirTest.absolutePath»/test_%d.csv''', trainSize, trainSize + testSize))
		val testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, -1, true, AlignmentMode.ALIGN_END)

		return new TrainAndTestData(trainData, testData, normalize(#[trainData, testData]))
	}

	@org.eclipse.xtend.lib.annotations.Data static class TrainAndTestData {
		DataSetIterator trainData
		DataSetIterator testData
		DataNormalization normalizer
	}

	@org.eclipse.xtend.lib.annotations.Data public static class PrepareData extends ActionTimed<DataFormat> {

		val int numberOfTimesteps
		val int miniBatchSize
		val List<String> rawStrings
		val int numOfVariables
		val int trainSize

		new(String rawDataFile, int numberOfTimesteps, int miniBatchSize) {
			this.numberOfTimesteps = numberOfTimesteps
			this.miniBatchSize = miniBatchSize
			rawStrings = Files.readAllLines(new File(baseDir, rawDataFile).toPath)
			numOfVariables = rawStrings.numOfVariables
			val trainingLines = rawStrings.size() - numberOfTimesteps - numberOfTimesteps - numberOfTimesteps
			trainSize = trainingLines - (trainingLines % miniBatchSize)
		}

		override getWorkloadSize() {
			trainSize + numberOfTimesteps
		}

		override call() throws Exception {
			featuresDirTrain.clean()
			labelsDirTrain.clean()
			featuresDirTest.clean()
			labelsDirTest.clean()
			message = "Creating training data"
			(0 .. trainSize).toList.parallelStream.forEach [
				val featuresPath = Paths.get('''«featuresDirTrain.absolutePath»/train_«it».csv''')
				val labelsPath = Paths.get('''«labelsDirTrain.absolutePath»/train_«it».csv''')
				(0 .. numberOfTimesteps).forEach [ step |
					Files.write(featuresPath, rawStrings.get(it + step).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
				]
				Files.write(labelsPath, rawStrings.get(it + numberOfTimesteps).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
				progress()
			]

			message = "Creating testing data"
			(trainSize .. (numberOfTimesteps + trainSize)).toList.parallelStream.forEach [
				val featuresPath = Paths.get('''«featuresDirTest.absolutePath»/test_«it».csv''')
				val labelsPath = Paths.get('''«labelsDirTest.absolutePath»/test_«it».csv''')
				(0 .. numberOfTimesteps).forEach [ step |
					Files.write(featuresPath, rawStrings.get(it + step).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
				]
				Files.write(labelsPath, rawStrings.get(it + numberOfTimesteps).concat("\n").bytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
				progress()
			]

			val trainingArray = createIndArrayFromStringList(rawStrings, numOfVariables, 0, trainSize)
			val testingArray = createIndArrayFromStringList(rawStrings, numOfVariables, trainSize, numberOfTimesteps)

			return new DataFormat(trainSize, numberOfTimesteps, numberOfTimesteps, numOfVariables, miniBatchSize, trainingArray, testingArray)
		}

	}

	@org.eclipse.xtend.lib.annotations.Data
	static class DataFormat {

		val int trainSize
		val int testSize
		val int numberOfTimesteps
		val int numOfVariables
		val int miniBatchSize
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
