package com.sirolf2009.dl4j.rnn.model

import java.util.LinkedList
import org.eclipse.xtend.lib.annotations.Data
import java.util.Collection

@Data class TimeSeries extends LinkedList<Point> {
	
	private val String name
	
	new(String name) {
		super()
		this.name = name
	}
	
	new(String name, Collection<Point> points) {
		super(points)
		this.name = name
	}
	
}
