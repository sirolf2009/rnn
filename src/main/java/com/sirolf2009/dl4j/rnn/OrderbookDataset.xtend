package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.model.LimitOrder
import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import java.util.List
import java.util.stream.Collectors
import java.util.stream.Stream
import org.ektorp.http.StdHttpClient
import org.ektorp.impl.StdCouchDbConnector
import org.ektorp.impl.StdCouchDbInstance

class OrderbookDataset {

	def static void main(String[] args) {
		val client = new StdHttpClient.Builder().url("http://172.17.0.1:5984").build()
		val instance = new StdCouchDbInstance(client)
		val orderbook = new StdCouchDbConnector("orderbook", instance)

		val writer = new PrintWriter(new FileOutputStream(new File("output.csv")))
		OrderbookDataset.fromOrders(orderbook.allDocIds.map[orderbook.get(LimitOrder, it)].toList(), 15).forEach[
			val line = it.map[price+","+amount].reduce[a,b|a+", "+b]
			writer.println(line)
			println(line)
		]
	}

	def static fromOrders(List<LimitOrder> orders, int cols) {
		orders.groupBy[date].entrySet.filter[value.filter[side.equals("ASK")].size > 0].map [
			val asks = value.stream.filter[side.equals("ASK")].sorted[a, b|a.price.compareTo(b.price)].limit(cols)
			val bids = value.stream.filter[side.equals("BID")].sorted[a, b|b.price.compareTo(a.price)].limit(cols)
			return Stream.concat(bids, asks).collect(Collectors.toList)
		]
	}

}
