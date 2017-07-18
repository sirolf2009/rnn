package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.model.LimitOrder
import java.util.List

class OrderbookDataset {
	
	def static fromOrders(List<LimitOrder> orders, double range, int cols) {
		orders.groupBy[date].entrySet.filter[value.filter[side.equals("ASK")].size > 0].map[
			val minAsk = value.filter[side.equals("ASK")].min[a,b|a.price.compareTo(b.price)].price
			val withinRange = value.filter[Math.abs(minAsk - price) <= range].toList()
			val ordersByCol = withinRange.groupBy[Math.floor(Math.abs(minAsk - range) / (range / cols))]
			ordersByCol.mapValues[map[price -> amount].reduce[a,b|
				Math.min(a.key, b.key) -> a.value + b.value
			]]
		]
	}
	
}