package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.indicator.RnnCloseIndicator
import eu.verdelhan.ta4j.Order.OrderType
import eu.verdelhan.ta4j.Strategy
import eu.verdelhan.ta4j.analysis.criteria.TotalProfitCriterion
import eu.verdelhan.ta4j.indicators.simple.ClosePriceIndicator
import eu.verdelhan.ta4j.trading.rules.OverIndicatorRule
import eu.verdelhan.ta4j.trading.rules.UnderIndicatorRule
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import java.time.Duration

@org.eclipse.xtend.lib.annotations.Data
class ScoreCalculatorBitstamp implements ScoreCalculator<MultiLayerNetwork> {

	val int numberOfTimesteps
	val series = CsvTradesLoader.loadBitstampSeries(Duration.ofMinutes(1))

	override calculateScore(MultiLayerNetwork net) {
		net.rnnClearPreviousState()
		val indicator = new RnnCloseIndicator(series, net, numberOfTimesteps)
		val backTestLong = net.backtestLong(indicator, numberOfTimesteps)
		val backTestShort = net.backtestShort(indicator, numberOfTimesteps)
		val profitLong = new TotalProfitCriterion().calculate(series, backTestLong);
		val profitShort = new TotalProfitCriterion().calculate(series, backTestShort);
		println("Profit Long : " + profitLong+" over "+backTestLong.tradeCount+" trades")
		println("Profit Short: " + profitShort+" over "+backTestLong.tradeCount+" trades")
		net.rnnClearPreviousState()
		return 1 / ((profitLong+profitShort)/2)
	}

	def static backtestLong(MultiLayerNetwork net, RnnCloseIndicator indicator, int forward) {
		val closePrice = new ClosePriceIndicator(indicator.timeSeries)
		val entryRule = new UnderIndicatorRule(closePrice, indicator)
		val exitRule = new OverIndicatorRule(closePrice, indicator)
		val strategy = new Strategy(entryRule, exitRule)
		return indicator.timeSeries.run(strategy)
	}

	def static backtestShort(MultiLayerNetwork net, RnnCloseIndicator indicator, int forward) {
		val closePrice = new ClosePriceIndicator(indicator.timeSeries)
		val exitRule = new UnderIndicatorRule(closePrice, indicator)
		val entryRule = new OverIndicatorRule(closePrice, indicator)
		val strategy = new Strategy(entryRule, exitRule)
		return indicator.timeSeries.run(strategy, OrderType.SELL)
	}

}
