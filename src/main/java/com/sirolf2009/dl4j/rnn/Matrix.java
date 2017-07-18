package com.sirolf2009.dl4j.rnn;

import org.eclipse.xtext.xbase.lib.Inline;

public class Matrix {
	
	@Inline("new double[$1][$2]")
	public static double[][] new2DDoubleArrayOfSize(int outerSize, int innerSize) {
		throw new UnsupportedOperationException();
	}

}
