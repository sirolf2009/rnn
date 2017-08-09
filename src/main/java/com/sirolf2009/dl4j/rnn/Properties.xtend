package com.sirolf2009.dl4j.rnn

import java.io.File

class Properties {
	
	public static val baseDir = new File("data")
	public static val featuresDirTrain = new File(baseDir, "features_train")
	public static val labelsDirTrain = new File(baseDir, "labels_train")
	public static val featuresDirTest = new File(baseDir, "features_test")
	public static val labelsDirTest = new File(baseDir, "labels_test")
	
}