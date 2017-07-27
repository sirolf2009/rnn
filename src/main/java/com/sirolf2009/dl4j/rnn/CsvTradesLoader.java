package com.sirolf2009.dl4j.rnn;

import eu.verdelhan.ta4j.Tick;
import eu.verdelhan.ta4j.TimeSeries;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.opencsv.CSVReader;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZonedDateTime;

/**
 * This class build a Ta4j time series from a CSV file containing trades.
 */
public class CsvTradesLoader {

	/**
	 * @return a time series from Bitstamp (bitcoin exchange) trades
	 */
	public static TimeSeries loadBitstampSeries(Duration interval) {
		try {
			// Reading all lines of the CSV file
			InputStream stream = new FileInputStream("data/bitstamp_trades_from_20131125_usd.csv");
			CSVReader csvReader = null;
			List<String[]> lines = null;
			try {
				csvReader = new CSVReader(new InputStreamReader(stream, Charset.forName("UTF-8")), ',');
				lines = csvReader.readAll();
				lines.remove(0); // Removing header line
			} catch (IOException ioe) {
				Logger.getLogger(CsvTradesLoader.class.getName()).log(Level.SEVERE, "Unable to load trades from CSV",
						ioe);
			} finally {
				if (csvReader != null) {
					try {
						csvReader.close();
					} catch (IOException ioe) {
					}
				}
			}

			if ((lines != null) && !lines.isEmpty()) {

				// Getting the first and last trades timestamps
				ZonedDateTime beginTime = ZonedDateTime.ofInstant(
						Instant.ofEpochMilli(Long.parseLong(lines.get(0)[0]) * 1000), ZoneId.systemDefault());
				ZonedDateTime endTime = ZonedDateTime.ofInstant(
						Instant.ofEpochMilli(Long.parseLong(lines.get(lines.size() - 1)[0]) * 1000),
						ZoneId.systemDefault());
				if (beginTime.isAfter(endTime)) {
					Instant beginInstant = beginTime.toInstant();
					Instant endInstant = endTime.toInstant();
					beginTime = ZonedDateTime.ofInstant(endInstant, ZoneId.systemDefault());
					endTime = ZonedDateTime.ofInstant(beginInstant, ZoneId.systemDefault());
					// Since the CSV file has the most recent trades at the top of the file, we'll
					// reverse the list to feed the List<Tick> correctly.
					Collections.reverse(lines);
				}
				// Building the empty ticks (every 300 seconds, yeah welcome in Bitcoin world)
				List<Tick> ticks = buildEmptyTicks(beginTime, endTime, (int) interval.getSeconds());
				// Filling the ticks with trades
				lines.parallelStream().forEach(tradeLine -> {
					ZonedDateTime tradeTimestamp = ZonedDateTime.ofInstant(
							Instant.ofEpochMilli(Long.parseLong(tradeLine[0]) * 1000), ZoneId.systemDefault());
					ticks.parallelStream().forEach(tick -> {
						if (tick.inPeriod(tradeTimestamp)) {
							double tradePrice = Double.parseDouble(tradeLine[1]);
							double tradeAmount = Double.parseDouble(tradeLine[2]);
							tick.addTrade(tradeAmount, tradePrice);
						}
					});
				});
//				for (String[] tradeLine : lines) {
//					ZonedDateTime tradeTimestamp = ZonedDateTime.ofInstant(
//							Instant.ofEpochMilli(Long.parseLong(tradeLine[0]) * 1000), ZoneId.systemDefault());
//					for (Tick tick : ticks) {
//						if (tick.inPeriod(tradeTimestamp)) {
//							double tradePrice = Double.parseDouble(tradeLine[1]);
//							double tradeAmount = Double.parseDouble(tradeLine[2]);
//							tick.addTrade(tradeAmount, tradePrice);
//						}
//					}
//				}
				// Removing still empty ticks
				removeEmptyTicks(ticks);
				return new TimeSeries("bitstamp_trades", ticks);
			}

			throw new IllegalArgumentException("An empty file has been provided!");
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Builds a list of empty ticks.
	 * 
	 * @param beginTime
	 *            the begin time of the whole period
	 * @param endTime
	 *            the end time of the whole period
	 * @param duration
	 *            the tick duration (in seconds)
	 * @return the list of empty ticks
	 */
	private static List<Tick> buildEmptyTicks(ZonedDateTime beginTime, ZonedDateTime endTime, int duration) {

		List<Tick> emptyTicks = new ArrayList<>();

		Duration tickDuration = Duration.ofSeconds(duration);
		ZonedDateTime tickEndTime = beginTime;
		do {
			tickEndTime = tickEndTime.plus(tickDuration);
			emptyTicks.add(new Tick(tickDuration, tickEndTime));
		} while (tickEndTime.isBefore(endTime));

		return emptyTicks;
	}

	/**
	 * Removes all empty (i.e. with no trade) ticks of the list.
	 * 
	 * @param ticks
	 *            a list of ticks
	 */
	private static void removeEmptyTicks(List<Tick> ticks) {
		for (int i = ticks.size() - 1; i >= 0; i--) {
			if (ticks.get(i).getTrades() == 0) {
				ticks.remove(i);
			}
		}
	}

	public static void main(String[] args) {
		TimeSeries series = CsvTradesLoader.loadBitstampSeries(Duration.ofMinutes(1));

		System.out.println("Series: " + series.getName() + " (" + series.getSeriesPeriodDescription() + ")");
		System.out.println("Number of ticks: " + series.getTickCount());
		System.out
				.println("First tick: \n" + "\tVolume: " + series.getTick(0).getVolume() + "\n" + "\tNumber of trades: "
						+ series.getTick(0).getTrades() + "\n" + "\tClose price: " + series.getTick(0).getClosePrice());
	}
}