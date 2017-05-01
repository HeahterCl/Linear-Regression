package huangchl.linearregression;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import com.csvreader.CsvReader;
import com.csvreader.CsvWriter;
public class ReadCSV {
	private static double STEP = 0.099255; // learning rate
	private static int TIMES = 10000; 
	private static double DELTA = 0.0001;
	private static double LAMBDA = 0;

	public static void main(String arg[]) {
		int colNum = 2;
		List<double[]> vector = new ArrayList<>(); // testing set
		List<double[]> validation = new ArrayList<>(); // validation set
		List<double[]> answers = new ArrayList<>();

		// get column number && store x's values
		try {
			String csvFilePath = "./src/resources/save_train.csv";
			CsvReader reader = new CsvReader(csvFilePath, ',', Charset.forName("SJIS"));

			reader.readHeaders();

			while (reader.readRecord()) {
				colNum = reader.getValues().length;
				double[] x = new double[colNum];
				x[0] = 1;
				for (int i = 1; i < colNum; i++) {
					x[i] = Double.valueOf(reader.get(i));
				}
				vector.add(x);
			}

			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		// randomly divide
		divide(vector, validation, 0.1);
		
		// set coefficients of multivariate linear function
		double[] param = new double[colNum - 1];
		for (int i = 0; i < colNum - 1; i++) {
			param[i] = 0;
		}
		
		double diff = Integer.MAX_VALUE;
		double previousCost = Integer.MAX_VALUE;
		int currentIters = 0;
		while (diff > DELTA) {
//			if (currentIters % 1000 == 0) {
//				STEP *= 0.9;
//			}
			currentIters ++;
			param = calIteration(vector, param, LAMBDA);
			double currentCost = calCostFunc(vector, param, LAMBDA);
			System.out.printf("Lambda: %f, Iteration: %d, Cost: %f\n", LAMBDA, currentIters, currentCost);
			
			diff = Math.abs(currentCost - previousCost);
			previousCost = currentCost;
		}
		
		//validate
		System.out.printf("VALIDATION: Lambda: %f, Cost: %f\n", LAMBDA, calCostFunc(validation, param, LAMBDA));
		
		//calculate the answer of testing set
		List<double[]> matrix = new ArrayList<>();
		double[] answer;
		
		try {
			// read the values of x from testing file
			String testFilePath = "./src/resources/save_test.csv";
			CsvReader testReader = new CsvReader(testFilePath, ',', Charset.forName("UTF-8"));
			testReader.readHeaders();
			
			//store x
			while(testReader.readRecord()) {
				double[] x = new double[testReader.getValues().length];
				x[0] = 1;
				for (int i = 1; i < testReader.getValues().length; i++) {
					x[i] = Double.valueOf(testReader.get(i));
				}
				matrix.add(x);
			}
			
			// calculate
			answer = new double[matrix.size()];
			for (int i = 0; i < matrix.size(); i++) {
				answer[i] = 0;
				for (int j = 0; j < matrix.get(i).length; j++) {
					answer[i] += matrix.get(i)[j]*param[j];
				}
			}
			
			String answerFilePath = "./src/resources/answer.csv";
			CsvWriter writer = new CsvWriter(answerFilePath, ',', Charset.forName("SJIS"));
			
			// write Header
			String[] header = {"id", "reference"};
			writer.writeRecord(header);
			
			// write answers
			for (int i = 0; i < answer.length; i++) {
				String[] record = {String.valueOf(i), String.valueOf(answer[i])};
				writer.writeRecord(record);
			}
			
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// randomly divide testing set into two parts
	private static void divide(List<double[]> testingSet, List<double[]> validation, double rate) {
		// random divide
		int valSetNum = (int)Math.floor(testingSet.size()*rate);
		int[] valSet = new int[valSetNum];
		int j = 0;
		while (j < valSetNum) {
			Random random = new Random();
			int randNum = random.nextInt(testingSet.size());
			if (!Arrays.asList(valSet).contains(randNum)) {
				valSet[j] = randNum;
				j++;
			}
		}	
		Arrays.sort(valSet);
				
		// divide the vector into two parts at the rate of 9:1
		// testing set : 90% (vector)
		// validation set : 10% (validation)
		for (int i = 0; i < valSet.length; i++) {
			validation.add(testingSet.get(valSet[i]-i));
			testingSet.remove(valSet[i]-i);
		}
	}
	
	// calculate cost function
	private static double calCostFunc(List<double[]> arg, double[] coefficients, double labmda) {
		double cost = 0;
		Iterator<double[]> ite = arg.iterator();
		while (ite.hasNext()) {
			double[] tempX = ite.next();
			double function = 0;
			for (int i = 0; i < tempX.length - 1; i++) {
				function += tempX[i] * coefficients[i];
			}
			cost += Math.pow((function - tempX[tempX.length - 1]), 2.0);
		}
		cost = cost/(2.0*arg.size());
		for (int i = 0; i < coefficients.length; i++) {
			cost += labmda*coefficients[i]*coefficients[i];
		}
		return cost;
	}

	// calculate coefficients after iteration
	private static double[] calIteration(List<double[]> arg, double[] coefficients, double lambda) {
		// calculate the sum
		double[] sum = new double[arg.size()];
		for (int i = 0; i < arg.size(); i++) {
			sum[i] = 0;
			for (int j = 0; j < coefficients.length; j++) {
				sum[i] += arg.get(i)[j] * coefficients[j];
			}
			sum[i] -= arg.get(i)[coefficients.length];
		}

		// calculate the coefficients
		double[] coeff_temp = new double[coefficients.length];
		for (int i = 0; i < coefficients.length; i++) {
			coeff_temp[i] = 0;
			for (int j = 0; j < arg.size(); j++) {
				coeff_temp[i] += sum[j] * arg.get(j)[i];
			}
			coeff_temp[i] /= arg.size();
			coeff_temp[i] *= STEP;
			coeff_temp[i] = coefficients[i] - coeff_temp[i];
			if (i != 0) {
				coeff_temp[i] -= STEP*lambda/arg.size();
			}
		}

		return coeff_temp;
	}
}
