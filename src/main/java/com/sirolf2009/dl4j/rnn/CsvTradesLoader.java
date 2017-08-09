package com.sirolf2009.dl4j.rnn;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.opencsv.CSVReader;
import com.sirolf2009.progressbar.Action;

import eu.verdelhan.ta4j.Tick;
import eu.verdelhan.ta4j.TimeSeries;

public class CsvTradesLoader extends Action<TimeSeries> {

	private static final String FILE = "data/bitstamp_trades_from_20131125_usd.csv";

	private final Duration interval;

	public CsvTradesLoader(Duration interval) {
		this.interval = interval;
	}

	@Override
	public TimeSeries call() throws Exception {
		try {
			InputStream stream = new FileInputStream(FILE);
			try (CSVReader csvReader = new CSVReader(new InputStreamReader(stream, Charset.forName("UTF-8")), ',')) {
				List<String[]> lines = csvReader.readAll();
				lines.remove(0); // Removing header line

				if ((lines != null) && !lines.isEmpty()) {
					ZonedDateTime beginTime = ZonedDateTime.ofInstant(Instant.ofEpochMilli(Long.parseLong(lines.get(0)[0]) * 1000), ZoneId.systemDefault());
					ZonedDateTime endTime = ZonedDateTime.ofInstant(Instant.ofEpochMilli(Long.parseLong(lines.get(lines.size() - 1)[0]) * 1000), ZoneId.systemDefault());
					if (beginTime.isAfter(endTime)) {
						setMessage("Reversing list");
						Instant beginInstant = beginTime.toInstant();
						Instant endInstant = endTime.toInstant();
						beginTime = ZonedDateTime.ofInstant(endInstant, ZoneId.systemDefault());
						endTime = ZonedDateTime.ofInstant(beginInstant, ZoneId.systemDefault());
						// Since the CSV file has the most recent trades at the top of the file, we'll
						// reverse the list to feed the List<Tick> correctly.
						Collections.reverse(lines);
					}

					setMessage("Building empty list");
					List<Tick> ticks = buildEmptyTicks(beginTime, endTime, (int) interval.getSeconds());

					setMessage("Parsing");
					lines.parallelStream().forEach(tradeLine -> {
						ZonedDateTime tradeTimestamp = ZonedDateTime.ofInstant(Instant.ofEpochMilli(Long.parseLong(tradeLine[0]) * 1000), ZoneId.systemDefault());
						ticks.parallelStream().filter(tick -> tick.inPeriod(tradeTimestamp)).findFirst().ifPresent(tick -> {
							double tradePrice = Double.parseDouble(tradeLine[1]);
							double tradeAmount = Double.parseDouble(tradeLine[2]);
							tick.addTrade(tradeAmount, tradePrice);
						});
						progress();
					});
					setMessage("Removing empty ticks");
					removeEmptyTicks(ticks);
					return new TimeSeries("bitstamp_trades", ticks);
				}
			} catch (IOException ioe) {
				Logger.getLogger(CsvTradesLoader.class.getName()).log(Level.SEVERE, "Unable to load trades from CSV", ioe);
			}
			throw new IllegalArgumentException("An empty file has been provided!");
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

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

	private static void removeEmptyTicks(List<Tick> ticks) {
		for (int i = ticks.size() - 1; i >= 0; i--) {
			if (ticks.get(i).getTrades() == 0) {
				ticks.remove(i);
			}
		}
	}

	@Override
	public int getWorkloadSize() {
		return countLines(FILE);
	}

	public static int countLines(String filename) {
		try (InputStream is = new BufferedInputStream(new FileInputStream(filename))) {
			byte[] c = new byte[1024];
			int count = 0;
			int readChars = 0;
			boolean empty = true;
			while ((readChars = is.read(c)) != -1) {
				empty = false;
				for (int i = 0; i < readChars; ++i) {
					if (c[i] == '\n') {
						++count;
					}
				}
			}
			return (count == 0 && !empty) ? 1 : count;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}