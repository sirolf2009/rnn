package com.sirolf2009.dl4j.rnn.model

import org.eclipse.xtend.lib.annotations.Data
import java.util.ArrayList

@Data class Dataset extends ArrayList<TimeSeries> {
	
	def getTimeSeries(String name) {
		parallelStream().filter[it.name.equals(name)].findFirst
	}
	
}
