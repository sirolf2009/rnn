package com.sirolf2009.dl4j.rnn.model

import java.util.ArrayList
import java.util.stream.Collectors
import org.eclipse.xtend.lib.annotations.EqualsHashCode
import org.eclipse.xtend.lib.annotations.ToString

@EqualsHashCode @ToString class Dataset extends ArrayList<TimeSeries> {
	
	def getTimeSeries(String name) {
		parallelStream().filter[it.name.equals(name)].findFirst
	}
	
	def subset(int indexFrom, int indexTo) {
		val dataset = new Dataset()
		dataset += stream.map[new TimeSeries(name, subList(indexFrom, indexTo))].collect(Collectors.toList())
		dataset
	}
	
}
