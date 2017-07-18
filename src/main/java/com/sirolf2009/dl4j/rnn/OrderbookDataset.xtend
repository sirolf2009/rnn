package com.sirolf2009.dl4j.rnn

import org.ektorp.http.StdHttpClient
import org.ektorp.impl.StdCouchDbInstance
import org.ektorp.impl.StdCouchDbConnector
import com.sirolf2009.dl4j.rnn.model.LimitOrder

class OrderbookDataset {
	
	def static void main(String[] args) {
		val client = new StdHttpClient.Builder().url("http://localhost:5984").username("admin").password("SimplySimplicity").build()
		val instance = new StdCouchDbInstance(client)
		val database = new StdCouchDbConnector("orderbook", instance)
		database.allDocIds.forEach[
			println(database.get(LimitOrder, it))
		]
	}
	
}