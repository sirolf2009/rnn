package com.sirolf2009.dl4j.rnn

import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import java.io.File
import java.nio.file.Files
import com.sirolf2009.dl4j.rnn.model.LimitOrder
import java.text.SimpleDateFormat
import java.util.stream.Collectors

class DbDump {

	def static getOrders() {
//		val file = new File("C:\\Users\\Floris\\Downloads\\db.json")
		val file = new File("/home/sirolf2009/Downloads/db.json")
		val gson = new GsonBuilder().create()
		return Files.lines(file.toPath).parallel().filter[startsWith("{\"id\":\"")].map [
			try {
				val document = gson.fromJson(substring(0, length - 1), JsonObject).getAsJsonObject("doc")
				val order = new LimitOrder() => [
					amount = document.getAsJsonPrimitive("amount").asDouble
					date = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSX").parse(document.getAsJsonPrimitive("date").asString)
					pair = document.getAsJsonPrimitive("pair").asString
					exchange = document.getAsJsonPrimitive("exchange").asString
					price = document.getAsJsonPrimitive("price").asDouble
					side = document.getAsJsonPrimitive("side").asString
				]
				return order
			} catch(Exception e) {
				return null
			}
		].filter[it !== null].filter[pair.equals("BTC/USD")].collect(Collectors.toList())
	}

}
