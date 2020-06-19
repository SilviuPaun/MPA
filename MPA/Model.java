package MPA;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.special.Gamma;

public class Model {
	
	private int I; //the number of mentions
	private int J; //number of annotators
	private int K; //number of classes
	private int[] Mi; //number of labels per mention
	private int[] Ni; //number of decisions per mention
	
	//variational parameters - same notation as in the paper
	private double[] lambda, eta;
	private double[][] gamma, mu;
	private double[][] theta, epsilon;
	
	private double[][] phi, zeta;
	
	//hyper-parameters
	private double a,b,d,e,t,u;
	
	//useful speed params
	private double[][] log_phi, log_zeta;
	
	private double[] digamma_lambda, digamma_eta; 
	private double[] digamma_lambdaPLUSeta; 
	private double[] lambdaPLUSeta;
	
	private double[][] digamma_gamma, digamma_mu; 
	private double[][] digamma_gammaPLUSmu; 
	private double[][] gammaPLUSmu;
	
	private double[][] digamma_theta, digamma_epsilon; 
	private double[][] digamma_thetaPLUSepsilon; 
	private double[][] thetaPLUSepsilon;
	
	//sufficient statistics
	double[][] ss_gamma, ss_mu;
	double[][] ss_theta, ss_epsilon;
	double[] ss_lambda, ss_eta;
	
	//data collections
	private List<String> items;
	private List<String> contexts; //the classes the labels are part of; e.g.: DN, DO
	private Map<String, List<String>> itemClasses; //the mention-level labels; e.g.: the antecedents
	private List<String> annotators;
	
	private Map<Integer, List<Integer>> itemContexts; // the z_i,m
	private Map<Pair, List<Integer>> itemClassAnnotations; // the y_i,m,n
	private Map<Integer, List<Integer>> itemAnnotations;
	private Map<Pair, Integer> annotatorOfItemAnnotation; // for jj(i,m,n)
	
	private Map<String, String> itemGoldStandard; //the gold labels for each mention
	
	//general settings
	private int RANDOM_SEED;
	private double convergence_threshold = Math.pow(10, -6);
	private String convergenceInfo;
	private Double runTime;
	
	public void setSeed(Integer seed) {
		RANDOM_SEED = seed;
	}
	
	public Model() {}
	
	public void Launch()
	{
		String relativePath = "C:/EMNLP2018/";
		String annotationsFile = "example.csv";
		
		LoadAnnotations(relativePath + annotationsFile);
		ProcessAnnotations();
		
		InitializeParameterSpace();
		InitializeParameters();
		RunModel();
		
		System.out.println("Processed file: " + annotationsFile);
		System.out.println(I + " items");
		System.out.println(J + " annotators");
		System.out.println(K + " classes: " + contexts);
		
		Map<Integer, Integer> inferredLabels = GetInferredLabels();
		double accuracy = ComputeAccuracy(inferredLabels);
		System.out.println("MPA accuracy: " + accuracy + " " + (int)(accuracy*I) + "/" + I);
		
		for(int seed = 1; seed<=10; seed++)
		{
			Random rand = new Random(seed);
			
			Map<Integer, Integer> majorityVote = GetMajorityVoting(rand);
			
			double accuracyMV = ComputeAccuracy(majorityVote);
			System.out.println("MV seed " + seed + " :" + accuracyMV + " " + (int)(accuracyMV*I) + "/" + I);
		}
	}
	
	private double ComputeAccuracy(Map<Integer, Integer> results) 
	{
		double matches = 0.0;
		for(int i=0; i<I; i++)
		{
			List<String> itemClassesList = itemClasses.get(items.get(i));
			
			String silverLabel = itemClassesList.get(results.get(i));
			String goldLabel = itemGoldStandard.get(items.get(i));
			
			if(silverLabel.equals(goldLabel))
				matches ++;
		}
		matches = matches / I;
		return matches;
	}

	private Map<Integer, Integer> GetMajorityVoting(Random rand) 
	{
		Map<Integer, Integer> majorityVoting = new HashMap<Integer, Integer>();
		
		for(int i=0; i<I; i++)
		{
			List<Integer> itemAnnotationsList = itemAnnotations.get(i);
			
			int Ni = itemAnnotationsList.size();
			
			Map<Integer, Integer> majorityVote = new HashMap<Integer, Integer>();
			
			for(int n=0; n<Ni; n++)
			{
				int annotationIndex = itemAnnotationsList.get(n);
				majorityVote.put(annotationIndex, majorityVote.containsKey(annotationIndex) ? majorityVote.get(annotationIndex) + 1 : 1);
			}
			
			List<Integer> votes = new ArrayList<Integer>();
			votes.addAll(majorityVote.values());
			
			int maxVotes = Collections.max(votes);
			
			List<Integer> classesWithMV = new ArrayList<Integer>();
			for(int classIndex : majorityVote.keySet())
			{
				int classVote = majorityVote.get(classIndex);
				if(classVote == maxVotes)
					classesWithMV.add(classIndex);
			}
			
			int size = classesWithMV.size();
			int randomClass = classesWithMV.get(rand.nextInt(size));
			
			majorityVoting.put(i, randomClass);
		}
		
		return majorityVoting;
	}
	
	private Map<Integer, Integer> GetInferredLabels() 
	{
		Map<Integer, Integer> result = new HashMap<Integer, Integer>();
		for(int i=0; i<I; i++)
		{
			double maxProb = 0.0;
			int maxProbIndex = -1;
			
			for(int m=0; m<Mi[i]; m++)
			{
				if(phi[i][m] >= maxProb)
				{
					maxProb = phi[i][m];
					maxProbIndex = m;
				}
			}
			result.put(i, maxProbIndex);
		}
		
		return result;
	}
	
	
	private void RunModel() 
	{
		boolean convergence = false;
		int nIter = 1;
		double lowerBoundOLD = 0.0;
		double lowerBoundNEW = 0.0;
		long startTime = System.currentTimeMillis();
		
		while(!convergence)
		{
			UpdateParameters();
			
			if(nIter==1)
			{
				lowerBoundOLD = ComputeLowerBound();
				System.out.println(RANDOM_SEED + " - lowerbound at " + nIter + ": " + lowerBoundOLD);
				nIter++;
			}
			else
			{
				int skip = 1; //if you don't want to check every iteration for convergence
				if(nIter % skip == 0) 
				{
					lowerBoundNEW = ComputeLowerBound();
					System.out.println(RANDOM_SEED + " - lowerbound at " + nIter + ": " + lowerBoundNEW);
					convergence = CheckConvergence(lowerBoundOLD, lowerBoundNEW);
					if(!convergence)
						lowerBoundOLD = lowerBoundNEW;
					else
					{
						convergenceInfo = "Convergence achieved in " + nIter + " iterations with a lowerbound value of " + lowerBoundNEW;
						System.out.println(RANDOM_SEED + " - " + convergenceInfo);
						
						long endTime = System.currentTimeMillis();
						runTime =  (endTime-startTime) / 1000.0;
						System.out.println(RANDOM_SEED + " - Run time: " + runTime + " seconds");
					}
				}
				nIter++;
			}
		}
	}
	
	private void UpdateParameters() 
	{
		//-----------reset sufficient statistics-----------------
		ss_lambda = new double[K];
		ss_eta = new double[K];
		
		ss_gamma = new double[J][K];
		ss_mu = new double[J][K];
		
		ss_theta = new double[J][K];
		ss_epsilon = new double[J][K];
		//-------------------------------------------------------
		
		for(int i=0; i<I; i++)
		{
			List<Integer> itemContextsList = itemContexts.get(i);
			
			for(int m=0; m<Mi[i]; m++)
			{
				int contextIdx = itemContextsList.get(m); 
				List<Integer> itemClassAnnotationsList = itemClassAnnotations.get(new Pair(i, m));
				
				log_phi[i][m] = digamma_lambda[contextIdx] - digamma_lambdaPLUSeta[contextIdx];
				log_zeta[i][m] = digamma_eta[contextIdx] - digamma_lambdaPLUSeta[contextIdx];
				
				
				for(int n=0; n<Ni[i]; n++)
				{
					int annotatorIdx = annotatorOfItemAnnotation.get(new Pair(i, n));
					int annotation = itemClassAnnotationsList.get(n);
					
					if(annotation == 1)
					{
						log_phi[i][m] += digamma_gamma[annotatorIdx][contextIdx] - digamma_gammaPLUSmu[annotatorIdx][contextIdx];
						
						log_zeta[i][m] += digamma_epsilon[annotatorIdx][contextIdx] - digamma_thetaPLUSepsilon[annotatorIdx][contextIdx];
					}
					else //annot = 0
					{
						log_phi[i][m] += digamma_mu[annotatorIdx][contextIdx] - digamma_gammaPLUSmu[annotatorIdx][contextIdx];
						
						log_zeta[i][m] += digamma_theta[annotatorIdx][contextIdx] - digamma_thetaPLUSepsilon[annotatorIdx][contextIdx];
					}
				}
				
				//normalize
				double maxExp = log_phi[i][m] > log_zeta[i][m] ? log_phi[i][m] : log_zeta[i][m];
				double sumExp = Math.exp(log_phi[i][m] - maxExp) + Math.exp(log_zeta[i][m] - maxExp);
				
				phi[i][m] = Math.exp(log_phi[i][m] - maxExp - Math.log(sumExp));
				zeta[i][m] = Math.exp(log_zeta[i][m] - maxExp - Math.log(sumExp));
				
				//update sufficient statistics
				ss_lambda[contextIdx] += phi[i][m];
				ss_eta[contextIdx] += zeta[i][m];
				for(int n=0; n<Ni[i]; n++)
				{
					int annotatorIdx = annotatorOfItemAnnotation.get(new Pair(i, n));
					int annotation = itemClassAnnotationsList.get(n);
					
					if(annotation == 1)
					{
						ss_gamma[annotatorIdx][contextIdx] += phi[i][m];
						ss_epsilon[annotatorIdx][contextIdx] += zeta[i][m];
					}
					else //annot = 0
					{
						ss_mu[annotatorIdx][contextIdx] += phi[i][m];
						ss_theta[annotatorIdx][contextIdx] += zeta[i][m];
					}
				}
			}
		}
		
		for(int h=0; h<K; h++)
		{
			lambda[h] = a + ss_lambda[h];
			eta[h] = b + ss_eta[h];
			
			lambdaPLUSeta[h] = lambda[h] + eta[h];
			digamma_lambda[h] = Gamma.digamma(lambda[h]);
			digamma_eta[h] = Gamma.digamma(eta[h]);
			digamma_lambdaPLUSeta[h] = Gamma.digamma(lambdaPLUSeta[h]);
		}
		
		for(int j=0; j<J; j++)
		{
			for(int h=0; h<K; h++)
			{
				gamma[j][h] = d + ss_gamma[j][h];
				mu[j][h] = e + ss_mu[j][h];

				gammaPLUSmu[j][h] = gamma[j][h] + mu[j][h];
				digamma_gammaPLUSmu[j][h] = Gamma.digamma(gammaPLUSmu[j][h]);
				digamma_gamma[j][h] = Gamma.digamma(gamma[j][h]);
				digamma_mu[j][h] = Gamma.digamma(mu[j][h]);
				
				
				theta[j][h] = t + ss_theta[j][h];
				epsilon[j][h] = u + ss_epsilon[j][h];

				thetaPLUSepsilon[j][h] = theta[j][h] + epsilon[j][h];
				digamma_thetaPLUSepsilon[j][h] = Gamma.digamma(thetaPLUSepsilon[j][h]);
				digamma_theta[j][h] = Gamma.digamma(theta[j][h]);
				digamma_epsilon[j][h] = Gamma.digamma(epsilon[j][h]);
			}
		}
	}
	
	
	private double ComputeLowerBound() 
	{
		double lowerbound = 0.0;
		
		for(int i=0; i<I; i++)
		{
			List<Integer> itemContextsList = itemContexts.get(i);
			
			for(int m=0; m<Mi[i]; m++)
			{
				int contextIdx = itemContextsList.get(m); 
				List<Integer> itemClassAnnotationsList = itemClassAnnotations.get(new Pair(i, m));
				
				lowerbound += phi[i][m] * (digamma_lambda[contextIdx] - digamma_lambdaPLUSeta[contextIdx]);
				lowerbound += zeta[i][m] * (digamma_eta[contextIdx] - digamma_lambdaPLUSeta[contextIdx]);
				
				if(phi[i][m] > 0)
					lowerbound -= phi[i][m] * Math.log(phi[i][m]);
				if(zeta[i][m] > 0)
					lowerbound -= zeta[i][m] * Math.log(zeta[i][m]);
				
				for(int n=0; n<Ni[i]; n++)
				{
					int annotatorIdx = annotatorOfItemAnnotation.get(new Pair(i, n));
					int annotation = itemClassAnnotationsList.get(n);
					
					if(annotation == 1)
					{
						lowerbound += phi[i][m] * (digamma_gamma[annotatorIdx][contextIdx] - digamma_gammaPLUSmu[annotatorIdx][contextIdx]);
						
						lowerbound += zeta[i][m] * (digamma_epsilon[annotatorIdx][contextIdx] -digamma_thetaPLUSepsilon[annotatorIdx][contextIdx]);
					}
					else //annot = 0
					{
						lowerbound += phi[i][m] * (digamma_mu[annotatorIdx][contextIdx] - digamma_gammaPLUSmu[annotatorIdx][contextIdx]);
						
						lowerbound += zeta[i][m] * (digamma_theta[annotatorIdx][contextIdx] -digamma_thetaPLUSepsilon[annotatorIdx][contextIdx]);
					}
				}
			}
		}
		
		for(int h=0; h<K; h++)
		{
			lowerbound += (a - 1) * (digamma_lambda[h] - digamma_lambdaPLUSeta[h]);
			lowerbound += (b - 1) * (digamma_eta[h] - digamma_lambdaPLUSeta[h]);
			lowerbound += Gamma.logGamma(a + b);
			lowerbound -= Gamma.logGamma(a);
			lowerbound -= Gamma.logGamma(b);
			
			lowerbound -= (lambda[h] - 1) * (digamma_lambda[h] - digamma_lambdaPLUSeta[h]);
			lowerbound -= (eta[h] - 1) * (digamma_eta[h] - digamma_lambdaPLUSeta[h]);
			lowerbound -= Gamma.logGamma(lambdaPLUSeta[h]);
			lowerbound += Gamma.logGamma(lambda[h]);
			lowerbound += Gamma.logGamma(eta[h]);
		}
		
		for(int j=0; j<J; j++)
		{
			for(int h=0; h<K; h++)
			{
				lowerbound += (d - 1) * (digamma_gamma[j][h] - digamma_gammaPLUSmu[j][h]);
				lowerbound += (e - 1) * (digamma_mu[j][h] - digamma_gammaPLUSmu[j][h]);
				lowerbound += Gamma.logGamma(d + e);
				lowerbound -= Gamma.logGamma(d);
				lowerbound -= Gamma.logGamma(e);
				
				lowerbound += (t - 1) * (digamma_theta[j][h] - digamma_thetaPLUSepsilon[j][h]);
				lowerbound += (u - 1) * (digamma_epsilon[j][h] - digamma_thetaPLUSepsilon[j][h]);
				lowerbound += Gamma.logGamma(t + u);
				lowerbound -= Gamma.logGamma(t);
				lowerbound -= Gamma.logGamma(u);
				
				lowerbound -= (gamma[j][h] - 1) * (digamma_gamma[j][h] - digamma_gammaPLUSmu[j][h]);
				lowerbound -= (mu[j][h] - 1) * (digamma_mu[j][h] - digamma_gammaPLUSmu[j][h]);
				lowerbound -= Gamma.logGamma(gammaPLUSmu[j][h]);
				lowerbound += Gamma.logGamma(gamma[j][h]);
				lowerbound += Gamma.logGamma(mu[j][h]);
				
				lowerbound -= (theta[j][h] - 1) * (digamma_theta[j][h] - digamma_thetaPLUSepsilon[j][h]);
				lowerbound -= (epsilon[j][h] - 1) * (digamma_epsilon[j][h] - digamma_thetaPLUSepsilon[j][h]);
				lowerbound -= Gamma.logGamma(thetaPLUSepsilon[j][h]);
				lowerbound += Gamma.logGamma(theta[j][h]);
				lowerbound += Gamma.logGamma(epsilon[j][h]);
				
			}
		}
		
		return lowerbound;
	}
	
	private boolean CheckConvergence(double lowerBoundOLD, double lowerBoundNEW) 
	{
		if(lowerBoundOLD>lowerBoundNEW)
			System.out.println(RANDOM_SEED + " - Non-increasing lowerbound warning");
		return Math.abs(lowerBoundNEW-lowerBoundOLD) < convergence_threshold;
	}
	
	private void InitializeParameters()
	{
		//flat hyper-parameters
		a = b = d = e = t = u = 1;
		
		for(int h=0; h<K; h++)
		{
			lambda[h] = a;
			eta[h] = b;
			
			lambdaPLUSeta[h] = lambda[h] + eta[h];
			digamma_lambda[h] = Gamma.digamma(lambda[h]);
			digamma_eta[h] = Gamma.digamma(eta[h]);
			digamma_lambdaPLUSeta[h] = Gamma.digamma(lambdaPLUSeta[h]);
		}
		
		for(int j=0; j<J; j++)
		{
			for(int h=0; h<K; h++)
			{
				gamma[j][h] = d + 0.1; //add a bit of extra mass to sensitivity
				mu[j][h] = e;

				gammaPLUSmu[j][h] = gamma[j][h] + mu[j][h];
				digamma_gammaPLUSmu[j][h] = Gamma.digamma(gammaPLUSmu[j][h]);
				digamma_gamma[j][h] = Gamma.digamma(gamma[j][h]);
				digamma_mu[j][h] = Gamma.digamma(mu[j][h]);
				
				
				theta[j][h] = t + 0.1; //add a bit of extra mass to specificity
				epsilon[j][h] = u;

				thetaPLUSepsilon[j][h] = theta[j][h] + epsilon[j][h];
				digamma_thetaPLUSepsilon[j][h] = Gamma.digamma(thetaPLUSepsilon[j][h]);
				digamma_theta[j][h] = Gamma.digamma(theta[j][h]);
				digamma_epsilon[j][h] = Gamma.digamma(epsilon[j][h]);
			}
		}
	}
	
	private void InitializeParameterSpace() 
	{
		phi = new double[I][];
		zeta = new double[I][];
		log_phi = new double[I][];
		log_zeta = new double[I][];
		
		for(int i=0; i<I; i++)
		{
			phi[i] = new double[Mi[i]];
			log_phi[i] = new double[Mi[i]];
			
			zeta[i] = new double[Mi[i]];
			log_zeta[i] = new double[Mi[i]];
		}
		
		lambda = new double[K];
		eta = new double[K];
		lambdaPLUSeta = new double[K];
		digamma_lambda = new double[K];
		digamma_eta = new double[K];
		digamma_lambdaPLUSeta = new double[K];
		
		gamma = new double[J][K];
		mu = new double[J][K];
		gammaPLUSmu = new double[J][K];
		digamma_gamma = new double[J][K];
		digamma_mu = new double[J][K];
		digamma_gammaPLUSmu = new double[J][K];

		theta = new double[J][K];
		epsilon = new double[J][K];
		thetaPLUSepsilon = new double[J][K];
		digamma_theta = new double[J][K];
		digamma_epsilon = new double[J][K];
		digamma_thetaPLUSepsilon = new double[J][K];
	}
	
	private void LoadAnnotations(String path) 
	{
		annotators = new ArrayList<String>();
		items = new ArrayList<String>();
		itemClasses = new HashMap<String, List<String>>();	
		itemAnnotations = new HashMap<Integer, List<Integer>>();
		annotatorOfItemAnnotation = new HashMap<Pair, Integer>();
		itemGoldStandard = new HashMap<String, String>();

		//these additional collections will significantly speed up loading the data
		Set<String> annotatorsSet = new HashSet<String>();
		Map<String, Integer> annotatorsIndexMap = new HashMap<String, Integer>();
		Set<String> itemsSet = new HashSet<String>();
		Map<String, Integer> itemsIndexMap = new HashMap<String, Integer>();

		int lineCounter = 0;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
			String fileLine = reader.readLine();
			fileLine = reader.readLine(); //skip header
			while(fileLine != null)
			{
				lineCounter++;
				if(lineCounter%10000 == 0)
					System.out.println("processed lines: " + lineCounter);
				
				String[] data = fileLine.split(",");
				
				String item = data[0];
				String annotator = data[1];
				String gold = data[2];
				String responseClass = data[3];
				
				boolean added_coder = annotatorsSet.add(annotator);
				if(added_coder)
				{
					annotators.add(annotator);
					annotatorsIndexMap.put(annotator, annotators.size() - 1);
				}
				int annotatorIndex = annotatorsIndexMap.get(annotator);
				
				
				boolean added_item = itemsSet.add(item);
				if(added_item)
				{
					items.add(item);
					itemsIndexMap.put(item, items.size() - 1);
				}
				int itemIndex = itemsIndexMap.get(item);
					
				List<String> iClasses = itemClasses.containsKey(item) ? itemClasses.get(item) : new ArrayList<String>();
				if(!iClasses.contains(responseClass))
					iClasses.add(responseClass);
				itemClasses.put(item, iClasses);
				
				int responseIndex = iClasses.indexOf(responseClass);
				
				List<Integer> itemAnt = itemAnnotations.containsKey(itemIndex) ? itemAnnotations.get(itemIndex) : new ArrayList<Integer>();
				itemAnt.add(responseIndex);
				itemAnnotations.put(itemIndex, itemAnt);
					
				annotatorOfItemAnnotation.put(new Pair(itemIndex, itemAnt.size() - 1), annotatorIndex);
				itemGoldStandard.put(item, gold);
			
			fileLine = reader.readLine();
		}
		reader.close();
		
		I = items.size();
		J = annotators.size();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void ProcessAnnotations() 
	{
		Mi = new int[I];
		Ni = new int[I];
		
		contexts = new ArrayList<String>();
		itemClassAnnotations = new HashMap<Pair, List<Integer>>();
		itemContexts = new HashMap<Integer, List<Integer>>();
		for(int i=0; i<I; i++)
		{
			String item = items.get(i);
			List<String> itemClassesList = itemClasses.get(item);
			List<Integer> itemAnnotationsList = itemAnnotations.get(i);
			
			Mi[i] = itemClassesList.size();
			Ni[i] = itemAnnotationsList.size();
			
			for(int m=0; m<Mi[i]; m++)
			{
				List<Integer> itemClassAnnotationsList = new ArrayList<Integer>(); 
				
				String itemClass = itemClassesList.get(m);
				String itemClassContext = itemClass.substring(0, itemClass.indexOf("(")); //this is the class of the label
				
				if(!contexts.contains(itemClassContext))
					contexts.add(itemClassContext);
				
//				String itemClassContext = itemClass;
				int itemClassContextIdx = contexts.indexOf(itemClassContext); 
				
				List<Integer> itemContextsList = itemContexts.containsKey(i) ? itemContexts.get(i) : new ArrayList<Integer>();
				itemContextsList.add(itemClassContextIdx);
				itemContexts.put(i, itemContextsList);
				
				for(int n=0; n<Ni[i]; n++)
				{
					int annotationIdx = itemAnnotationsList.get(n);
					String annotationString = itemClassesList.get(annotationIdx);
					
					int y_imn = annotationString.equals(itemClass) ? 1 : 0;
					itemClassAnnotationsList.add(y_imn);
				}
				
				Pair itemClassPair = new Pair(i, m);
				itemClassAnnotations.put(itemClassPair, itemClassAnnotationsList);
			}
		}
		K = contexts.size();
	}
}
