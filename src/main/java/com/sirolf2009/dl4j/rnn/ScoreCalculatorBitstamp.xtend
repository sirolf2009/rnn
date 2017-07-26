package com.sirolf2009.dl4j.rnn

import com.sirolf2009.dl4j.rnn.indicator.RnnCloseIndicator
import eu.verdelhan.ta4j.Strategy
import eu.verdelhan.ta4j.analysis.criteria.TotalProfitCriterion
import eu.verdelhan.ta4j.indicators.simple.ClosePriceIndicator
import eu.verdelhan.ta4j.trading.rules.OverIndicatorRule
import eu.verdelhan.ta4j.trading.rules.UnderIndicatorRule
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import java.time.Duration

@org.eclipse.xtend.lib.annotations.Data class ScoreCalculatorBitstamp implements ScoreCalculator<MultiLayerNetwork> {

	val int numberOfTimesteps
	val series = CsvTradesLoader.loadBitstampSeries(Duration.ofMinutes(15))

	override calculateScore(MultiLayerNetwork net) {
		net.rnnClearPreviousState()
		val indicator = new RnnCloseIndicator(series, net, numberOfTimesteps)
		val closePrice = new ClosePriceIndicator(series)
		val entryRule = new UnderIndicatorRule(closePrice, indicator)
		val exitRule = new OverIndicatorRule(closePrice, indicator)
		val strategy = new Strategy(entryRule, exitRule)
		val record = series.run(strategy)
		val profit = new TotalProfitCriterion().calculate(series, record)
		println("Profit: "+profit)
		net.rnnClearPreviousState()
		return 1/profit
	}

}
