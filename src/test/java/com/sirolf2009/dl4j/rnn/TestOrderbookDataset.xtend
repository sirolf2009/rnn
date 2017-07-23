package com.sirolf2009.dl4j.rnn

import org.junit.Test

class TestOrderbookDataset {

	@Test
	def void test() {
		println("Loading db orders")
		val orders = DbDump.orders
		println('''Loaded «orders.size» orders''')
		val slices = OrderbookDataset.fromOrders(orders, 15).toList
		println('''Loaded «slices.size» slices''')
		slices.forEach [
			println(it.map[price+":"+amount].reduce[a,b|a+", "+b])
		]
	}

}
