package com.sirolf2009.dl4j.rnn.model

import java.util.Date
import org.eclipse.xtend.lib.annotations.Accessors
import org.eclipse.xtend.lib.annotations.EqualsHashCode
import org.eclipse.xtend.lib.annotations.ToString

@Accessors
@EqualsHashCode
@ToString
class LimitOrder {
	
	var String id
	var String revision
	var String side
	var Date date
	var String pair
	var String exchange
	var double price
	var double amount
	
}
