package com.sirolf2009.dl4j.rnn.model

import java.util.Date
import org.eclipse.xtend.lib.annotations.Data

@Data class Point {
	
	val Double value
	val Date time
	
	def +(Double other) {
		return new Point(value + other, time)
	}
	def -(Double other) {
		return new Point(value - other, time)
	}
	def *(Double other) {
		return new Point(value * other, time)
	}
	def /(Double other) {
		return new Point(value / other, time)
	}
	def >(Double other) {
		return value > other
	}
	def <(Double other) {
		return value < other
	}
	def >=(Double other) {
		return value >= other
	}
	def <=(Double other) {
		return value <= other
	}
	
}