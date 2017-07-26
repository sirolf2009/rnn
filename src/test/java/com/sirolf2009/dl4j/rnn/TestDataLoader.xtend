package com.sirolf2009.dl4j.rnn

import org.junit.Test

class TestDataLoader {
	
	@Test
	def void test() {
		println(DataLoader.loadOHLCV2017.firstTick)
		println(DataLoader.loadOHLCV2017.lastTick)
		println(DataLoader.loadOHLCV2017.tickCount)
	}
	
}