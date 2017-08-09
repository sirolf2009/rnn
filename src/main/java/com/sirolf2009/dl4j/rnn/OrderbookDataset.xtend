package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.model.LimitOrder
import java.util.List
import java.util.stream.Collectors
import java.util.stream.Stream

class OrderbookDataset {

	def static fromOrders(List<LimitOrder> orders, int cols) {
		orders.groupBy[date].entrySet.filter[value.filter[side.equals("ASK")].size > 0].map [
			val asks = value.stream.filter[side.equals("ASK")].sorted[a, b|a.price.compareTo(b.price)].limit(cols)
			val bids = value.stream.filter[side.equals("BID")].sorted[a, b|b.price.compareTo(a.price)].limit(cols)
			return Stream.concat(bids, asks).collect(Collectors.toList)
		]
	}

}
