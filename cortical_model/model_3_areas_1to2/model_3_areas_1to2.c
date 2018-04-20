
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>

/* Needed for building mex file*/
#include <mex.h>
#include <matrix.h>

/* Conventions for variable names:
/* -------------------------------
/* - prefix "e" stands for "excitatory"
/* - prefix "i" stands for "inhibitory"
/* - prefix "x" stands for "external"
/*
/* - prefix "a" stands for "AMPA"
/* - prefix "g" stands for "GABA"
/*
/* - prefix "N"   stands for "number of..."
/* - prefix "tot" stands for "total"
/*
/* - prefix "e2e" stands for "excitatory to excitatory"
/* - prefix "e2i" stands for "excitatory to inhibitory"
/* - prefix "x2e" stands for "external to excitatory"
/* - prefix "x2i" stands for "external to inhibitory"
/*
/* - "T"  stands for greek letter "tau"
/* - "NU" stands for greek letter "nu"
/* - "S"  stands for greek letter "sigma"
/*
/* - "nrn"  stands for "neurons"
/* - "FR"   stands for "firing-rate"
/* - "SP"   stands for "spike"
/* - "SPC"  stands for "spike-count"*/




void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])

{
	/*/ Variables declaration ----------------------------------------------

	/* Time:*/
	double  Dt;       /*/ time resolution*/
	mwIndex t;        /*/ time index (in Dt steps)*/
	mwSize  simulLen; /*/ length of the simulation (in Dt steps)*/


	mwSize  eCycSize; /*/ size of cycle for excitatory nerons of area 1*/
	mwSize  iCycSize; /*/ size of cycle for inhibitory nerons of area 1*/
	mwIndex eCycIndx; /*/ cycle index for excitatory neurons of area 1*/
	mwIndex iCycIndx; /*/ cycle index for inhibitory neurons of area 1*/
	mwIndex eCycSP;   /*/ arrival cycle of spike emitted by pre-syn. ex. nrn of area 1*/
	mwIndex iCycSP;   /*/ arrival cycle of spike emitted by pre-syn. in. nrn of area 1*/

	// same for area 2
	mwSize  eCycSize2;
	mwSize  iCycSize2;
	mwIndex eCycIndx2;
	mwIndex iCycIndx2;
	mwIndex eCycSP2;
	mwIndex iCycSP2;

	// same for area 3
	mwSize  eCycSize3;
	mwSize  iCycSize3;
	mwIndex eCycIndx3;
	mwIndex iCycIndx3;
	mwIndex eCycSP3;
	mwIndex iCycSP3;

	//same for interareal
	mwSize  eCycSizeinter;
	mwIndex eCycIndxinter;
	mwIndex iCycIndxinter;
	mwIndex eCycSPinter;
	mwIndex iCycSPinter;



	/*/ Number of neurons:*/
	mwSize  eNnrn;   /*/ number of excitatory neurons in one area*/
	mwSize  iNnrn;   /*/ number of inhibitory neurons in one area*/
	mwSize  totNnrn; /*/ total number of neurons in the network*/
	mwSize  totNnrnt; // third of it, total in an area
	mwIndex nrn;     /*/ neuron index*/
	mwIndex nrn1;    /*/    "     "*/
	mwIndex nrn2;    /*/    "     "*/

	/*/ Random numbers generation*/
	double RAND_MAX_double;
	double rndNum;
	int seed1;
	int seed2;
	int seed3;
	int seed4;
	int seed5; // Seed for the generation of the 3rd network


	/*/* Potentials, common to all neurons in the network*/
	double Vthr; // Voltage threshold for spiking
	double eVres;// Voltage of reset after spiking the excitatory neuron
	double iVres;//same inhibitory

	/*/ Time constants (area 1)*/
	double eTm;//membrane time constant excitatory
	double iTm;//same inhibitory

	double eTrp;//%% refractory period excitatory
	double iTrp;// same inhibitory

	double eTl;//time lag of spikes excitatory inside an area
	double iTl;// same inhibitory

	double eTlinter;// time lag between areas

	double e2eTr;//time constant of rise synaptic current of a spike
	double e2eTd;//time constant of decay synaptic current of a spike 
	double e2iTr;
	double e2iTd;

	double iTr;//for the inhibitory no distinction is made between i2e and i2i
	double iTd;

	//time constanta area 2, analogous to area 1

	double eTm2;
	double iTm2;

	double eTrp2;
	double iTrp2;

	double eTl2;
	double iTl2;

	double e2eTr2;
	double e2eTd2;
	double e2iTr2;
	double e2iTd2;

	double iTr2;
	double iTd2;

	//time constanta area 3, analogous to area 1

	double eTm3;
	double iTm3;

	double eTrp3;
	double iTrp3;

	double eTl3;
	double iTl3;

	double e2eTr3;
	double e2eTd3;
	double e2iTr3;
	double e2iTd3;

	double iTr3;
	double iTd3;



	/*/ Connectivities (synaptic efficacies), following the convention of names*/

	// area 1
	double i2iJ;
	double e2iJ;
	double x2iJ;
	double i2eJ;
	double e2eJ;
	double x2eJ;

	// area 2
	double i2iJ2;
	double e2iJ2;
	double x2iJ2;
	double i2eJ2;
	double e2eJ2;
	double x2eJ2;

	// area 3
	double i2iJ3;
	double e2iJ3;
	double x2iJ3;
	double i2eJ3;
	double e2eJ3;
	double x2eJ3;

	//interareal connectivity
	double e2eJ122; //synaptic efficacy from area 1 to area 2
	double e2iJ122;


	/*/ Connections: These are the connections from a neuron to all others, not the neurons which connect to a particular one, which is the inverse*/
	double   p;  // probability of connection
	mwSize  *A;  /*/ matrix storing neurons connections of area 1*/
	mwSize  *A2; // matrix connections on area 2
	mwSize  *A3; // matrix connections on area 2
	mwSize  *A122;  //same from 1 to 2
	mwSize  *Aei122;  //same from 1 to 2


	mwSize   a;    /*/ element of A*/
	mwIndex  con;  /*/ connection index*/
	mwSize  *Ncon; /*/ array storing the number of connections of each neuron of area 1*/
	mwSize  *Ncon2; // same area 2
	mwSize  *Ncon3; // same area 3
	mwSize  *Ncon122;  // same excitatory-excitatory from 1 to 2
	mwSize  *Neicon122; // same excitatory-inhibitory from 1 to 2


	/*/ Spikes*/
	double *tLastSP; /*/ Time of last fired spike (to apply the refractory period)*/

	/*/ Firing rates*/

	//area 1
	double *xFR;     /*/ input firing rate to each neuron*/
	double  exp_xFR; /*/ exp(xFR)*/
	double *eFR;     /*/ number of excitatory action potentials at each time*/
	double *iFR;     /*/ number of inhibitory action potentials at each time*/

	//area 2
	double *xFR2;     /*/ input firing rate to each neuron*/
	double  exp_xFR2; /*/ exp(xFR)*/
	double *eFR2;     /*/ number of excitatory action potentials at each time*/
	double *iFR2;     /*/ number of inhibitory action potentials at each time*/

	//area 3
	double *xFR3;     /*/ input firing rate to each neuron*/
	double  exp_xFR3; /*/ exp(xFR)*/
	double *eFR3;     /*/ number of excitatory action potentials at each time*/
	double *iFR3;     /*/ number of inhibitory action potentials at each time*/


	/*variables needed to integrate the differential equations*/


	//voltages (mazzoni 2008 eq. 1)
	double *V; //value of voltage of all neurons at a given time
	double V0;
	double  V_k1;
	double  V_k2;

	double C;

	///auxiliary variables X (mazzoni 2008 eqs 3 and 5)
	//ampa excitatory inside an area
	double *aX; //value of auxiliary ampa variable of all neurons at a given time
	double  aX0;
	double  aX_k1;
	double  aX_k2;

	/// external arriving to an area
	double *xX;
	double  xX0;
	double  xX_k1;
	double  xX_k2;

	////inhibitory gaba inside an area
	double *gX;
	double  gX0;
	double  gX_k1;
	double  gX_k2;

	// excitatory inter areal
	double *iX;
	double  iX0;
	double  iX_k1;
	double  iX_k2;

	//exitatory to inhibitory inter areal
	double *uX;
	double  uX0;
	double  uX_k1;
	double  uX_k2;

	////current variables

	//total current
	double  synI;
	double  synI_k1;

	///ampa currents inside an area
	double *aI;
	double  aI0;
	double  aI_k1;
	double  aI_k2;

	//interareal excitatory currents
	double *iI;
	double  iI0;
	double  iI_k1;
	double  iI_k2;

	//external current arriving to an area
	double *xI;
	double  xI0;
	double  xI_k1;
	double  xI_k2;

	///gaba inside an area
	double *gI;
	double  gI0;
	double  gI_k1;
	double  gI_k2;

	// inter areal excitatory to inhibitory
	double *uI;
	double  uI0;
	double  uI_k1;
	double  uI_k2;


	/*time series of the currents*/
	//In the current program only the currents arriving to excitatory neurons are saved to model the LFP

	//area 1
	double *e2eI;
	double *i2eI;
	//double *e2iI;
	//double *i2iI;
	double *x2eI;
	//double *x2iI;

	//area 2
	double *e2eI2;
	double *i2eI2;
	//double *e2iI2;
	//double *i2iI2;
	double *x2eI2;
	//double *x2iI2;

	//area 3
	double *e2eI3;
	double *i2eI3;
	//double *e2iI3;
	//double *i2iI3;
	double *x2eI3;
	//double *x2iI3;

	//inter areal
	double *e2eI122;
	double *e2iI122;

	//time series of the voltages
	//area 1
	double *Ve;
	double *Vi;

	//area 2
	double *Ve2;
	double *Vi2;

	//area 3
	double *Ve3;
	double *Vi3;


	/*/ Spike counts*/
	/* array with the number of spikes of one type arriving on each neuron from the network*/
	/*The first totNnrn elements represents the spike of one type arriving on each neurons at the first time step of the cycle*/
	//area 1
	double *eSPC;
	double *iSPC;
	//area 2
	double *eSPC2;
	double *iSPC2;
	//area 3
	double *eSPC3;
	double *iSPC3;
	//interareal
	double *eSPC122;
	double *eiSPC122;

	///variables for spike counts contributing to currents for a given neuron at a given time
	//area 1
	double  xSPC;
	double  e2eSPC;
	double  e2iSPC;
	double  i2eSPC;
	double  i2iSPC;
	//area 2
	double  xSPC2;
	double  e2eSPC2;
	double  e2iSPC2;
	double  i2eSPC2;
	double  i2iSPC2;
	//area 3
	double  xSPC3;
	double  e2eSPC3;
	double  e2iSPC3;
	double  i2eSPC3;
	double  i2iSPC3;

	// interareal
	double  e2eSPC122;
	double  e2iSPC122;

	///counters to get the firing rates
	int eCounter;
	int iCounter;
	int eCounter2;
	int iCounter2;
	int eCounter3;
	int iCounter3;

	/*/* Defining the network properties ------------------------------------*/
	//some network properties are defined here, others are as input to the mex function

	/*/* Connectivity, i.e., connection probability, p, is 20%:*/
	p = 0.2;

	/*/* When the membrane potential crosses a threshold, Vthr, of 18mV above
	/* the resting potential:*/
	Vthr = 18;
	/*/* The neuron fires following the following sequence:
	/* 1. The neuron potential is reset to a value Vres above resting
	/*    potential.*/
	eVres = 11;
	iVres = 11;
	/*/* 2. The neuron cannot fire again for a refractory period, Trp, equal
	/*    to 2ms for E neurons and 1ms for I neurons.*/
	eTrp = 2;
	iTrp = 1;
	eTrp2 = 2;
	iTrp2 = 1;
	eTrp3 = 2;
	iTrp3 = 1;


	/*Input variables to the function*/


	/*/* The time-step is provided as the first input:*/
	Dt = *mxGetPr(prhs[0]);
	simulLen = mxGetNumberOfElements(prhs[1]); //length of the time-series

	xFR = mxGetPr(prhs[1]);// external Poisson rate of area 1
	xFR2 = mxGetPr(prhs[2]);// external Poisson rate of area 2
	xFR3 = mxGetPr(prhs[3]);// external Poisson rate of area 3

	///seeds to generate random numbers
	seed1 = (int)*mxGetPr(prhs[4]);// for connectivity matrices Area 1
	seed2 = (int)*mxGetPr(prhs[5]); //for Poisson processes
	seed3 = (int)*mxGetPr(prhs[6]);// for connectivity matrices Area 2
	seed4 = (int)*mxGetPr(prhs[7]);// for connectivity matrices Interareal
	seed5 = (int)*mxGetPr(prhs[8]);// for connectivity matrices Area 3

	//inputs of time constants area 1
	eTm = *mxGetPr(prhs[9]);
	iTm = *mxGetPr(prhs[10]);
	eTl = *mxGetPr(prhs[11]);
	iTl = *mxGetPr(prhs[12]);
	e2eTr = *mxGetPr(prhs[13]);
	e2eTd = *mxGetPr(prhs[14]);
	e2iTr = *mxGetPr(prhs[15]);
	e2iTd = *mxGetPr(prhs[16]);
	iTr = *mxGetPr(prhs[17]);
	iTd = *mxGetPr(prhs[18]);

	//inputs of time constants area 2
	eTm2 = *mxGetPr(prhs[19]);
	iTm2 = *mxGetPr(prhs[20]);
	eTl2 = *mxGetPr(prhs[21]);
	iTl2 = *mxGetPr(prhs[22]);
	e2eTr2 = *mxGetPr(prhs[23]);
	e2eTd2 = *mxGetPr(prhs[24]);
	e2iTr2 = *mxGetPr(prhs[25]);
	e2iTd2 = *mxGetPr(prhs[26]);
	iTr2 = *mxGetPr(prhs[27]);
	iTd2 = *mxGetPr(prhs[28]);

	//inputs of time constants area 3
	eTm3 = *mxGetPr(prhs[29]);
	iTm3 = *mxGetPr(prhs[30]);
	eTl3 = *mxGetPr(prhs[31]);
	iTl3 = *mxGetPr(prhs[32]);
	e2eTr3 = *mxGetPr(prhs[33]);
	e2eTd3 = *mxGetPr(prhs[34]);
	e2iTr3 = *mxGetPr(prhs[35]);
	e2iTd3 = *mxGetPr(prhs[36]);
	iTr3 = *mxGetPr(prhs[37]);
	iTd3 = *mxGetPr(prhs[38]);

	//time constant lag interareal
	eTlinter = *mxGetPr(prhs[39]);

	///synaptic efficacies area 1
	i2iJ = *mxGetPr(prhs[40]);
	e2iJ = *mxGetPr(prhs[41]);
	x2iJ = *mxGetPr(prhs[42]);
	i2eJ = *mxGetPr(prhs[43]);
	e2eJ = *mxGetPr(prhs[44]);
	x2eJ = *mxGetPr(prhs[45]);

	///synaptic efficacies area 2
	i2iJ2 = *mxGetPr(prhs[46]);
	e2iJ2 = *mxGetPr(prhs[47]);
	x2iJ2 = *mxGetPr(prhs[48]);
	i2eJ2 = *mxGetPr(prhs[49]);
	e2eJ2 = *mxGetPr(prhs[50]);
	x2eJ2 = *mxGetPr(prhs[51]);

	///synaptic efficacies area 3
	i2iJ3 = *mxGetPr(prhs[52]);
	e2iJ3 = *mxGetPr(prhs[53]);
	x2iJ3 = *mxGetPr(prhs[54]);
	i2eJ3 = *mxGetPr(prhs[55]);
	e2eJ3 = *mxGetPr(prhs[56]);
	x2eJ3 = *mxGetPr(prhs[57]);

	///interareal efficacies
	e2eJ122 = *mxGetPr(prhs[58]);
	e2iJ122 = *mxGetPr(prhs[59]);

	//number of cells
	eNnrn = (mwSize)*mxGetPr(prhs[60]);
	iNnrn = (mwSize)*mxGetPr(prhs[61]);

	totNnrn = 3 * (eNnrn + iNnrn);
	totNnrnt = totNnrn / 3;

	/*/* Normalizing time in Dt units --------------------------------------------*/
	eTm = eTm / Dt;
	iTm = iTm / Dt;

	e2eTr = e2eTr / Dt;
	e2iTr = e2iTr / Dt;

	e2eTd = e2eTd / Dt;
	e2iTd = e2iTd / Dt;

	iTr = iTr / Dt;
	iTd = iTd / Dt;

	eTrp = eTrp / Dt;
	iTrp = iTrp / Dt;

	eTl = eTl / Dt;
	iTl = iTl / Dt;

	eTm2 = eTm2 / Dt;
	iTm2 = iTm2 / Dt;

	e2eTr2 = e2eTr2 / Dt;
	e2iTr2 = e2iTr2 / Dt;

	e2eTd2 = e2eTd2 / Dt;
	e2iTd2 = e2iTd2 / Dt;

	iTr2 = iTr2 / Dt;
	iTd2 = iTd2 / Dt;

	eTrp2 = eTrp2 / Dt;
	iTrp2 = iTrp2 / Dt;

	eTl2 = eTl2 / Dt;
	iTl2 = iTl2 / Dt;

	eTm3 = eTm3 / Dt;
	iTm3 = iTm3 / Dt;

	e2eTr3 = e2eTr3 / Dt;
	e2iTr3 = e2iTr3 / Dt;

	e2eTd3 = e2eTd3 / Dt;
	e2iTd3 = e2iTd3 / Dt;

	iTr3 = iTr3 / Dt;
	iTd3 = iTd3 / Dt;

	eTrp3 = eTrp3 / Dt;
	iTrp3 = iTrp3 / Dt;

	eTl3 = eTl3 / Dt;
	iTl3 = iTl3 / Dt;

	eTlinter = eTlinter / Dt;

	///fixes length of the cycles to keep track of the currents given the time lag

	eCycSize = (mwSize)(eTl)+1;
	iCycSize = (mwSize)(iTl)+1;

	eCycSize2 = (mwSize)(eTl2)+1;
	iCycSize2 = (mwSize)(iTl2)+1;

	eCycSize3 = (mwSize)(eTl3)+1;
	iCycSize3 = (mwSize)(iTl3)+1;

	eCycSizeinter = (mwSize)(eTlinter)+1;

	/* Assigning ouputs ---------------------------------------------------*/
	plhs[0] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	e2eI = mxGetPr(plhs[0]);

	plhs[1] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	i2eI = mxGetPr(plhs[1]);

	plhs[2] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	eFR = mxGetPr(plhs[2]);

	plhs[3] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	iFR = mxGetPr(plhs[3]);

	//plhs[4] = mxCreateDoubleMatrix(simulLen, 1, mxREAL); 
	//e2iI = mxGetPr(plhs[4]); 

	//plhs[5] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	//i2iI = mxGetPr(plhs[5]);

	plhs[4] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	x2eI = mxGetPr(plhs[4]);

	//plhs[7] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	//x2iI = mxGetPr(plhs[7]);

	plhs[5] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	Ve = mxGetPr(plhs[5]);

	plhs[6] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	Vi = mxGetPr(plhs[6]);

	// Area 2

	plhs[7] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	e2eI2 = mxGetPr(plhs[7]);

	plhs[8] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	i2eI2 = mxGetPr(plhs[8]);

	plhs[9] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	eFR2 = mxGetPr(plhs[9]);

	plhs[10] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	iFR2 = mxGetPr(plhs[10]);

	//plhs[16] = mxCreateDoubleMatrix(simulLen, 1, mxREAL); 
	//e2iI2 = mxGetPr(plhs[16]); 

	//plhs[17] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	//i2iI2 = mxGetPr(plhs[17]);

	plhs[11] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	x2eI2 = mxGetPr(plhs[11]);

	//plhs[19] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	//x2iI2 = mxGetPr(plhs[19]);

	plhs[12] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	Ve2 = mxGetPr(plhs[12]);

	plhs[13] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	Vi2 = mxGetPr(plhs[13]);

	// Area 3

	plhs[14] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	e2eI3 = mxGetPr(plhs[14]);

	plhs[15] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	i2eI3 = mxGetPr(plhs[15]);

	plhs[16] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	eFR3 = mxGetPr(plhs[16]);

	plhs[17] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	iFR3 = mxGetPr(plhs[17]);

	//plhs[16] = mxCreateDoubleMatrix(simulLen, 1, mxREAL); 
	//e2iI2 = mxGetPr(plhs[16]); 

	//plhs[17] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	//i2iI2 = mxGetPr(plhs[17]);

	plhs[18] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	x2eI3 = mxGetPr(plhs[18]);

	//plhs[19] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	//x2iI2 = mxGetPr(plhs[19]);

	plhs[19] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	Ve3 = mxGetPr(plhs[19]);

	plhs[20] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	Vi3 = mxGetPr(plhs[20]);

	// Interareal

	plhs[21] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	e2eI122 = mxGetPr(plhs[21]);

	plhs[22] = mxCreateDoubleMatrix(simulLen, 1, mxREAL);
	e2iI122 = mxGetPr(plhs[22]);


	/* Allocating arrays: NB mxCalloc initializes the allocated memory to 0.*/
	tLastSP = mxCalloc(totNnrn, sizeof(double)); /* array with the time of spiking*/

	aX = mxCalloc(totNnrn, sizeof(double));
	gX = mxCalloc(totNnrn, sizeof(double));
	xX = mxCalloc(totNnrn, sizeof(double));
	iX = mxCalloc(totNnrn, sizeof(double));
	uX = mxCalloc(totNnrn, sizeof(double));


	aI = mxCalloc(totNnrn, sizeof(double));
	gI = mxCalloc(totNnrn, sizeof(double));
	xI = mxCalloc(totNnrn, sizeof(double));
	iI = mxCalloc(totNnrn, sizeof(double));
	uI = mxCalloc(totNnrn, sizeof(double));



	V = mxCalloc(totNnrn, sizeof(double));
	Ncon = mxCalloc(totNnrnt, sizeof(mwSize));
	Ncon2 = mxCalloc(totNnrnt, sizeof(mwSize));
	Ncon3 = mxCalloc(totNnrnt, sizeof(mwSize));
	Ncon122 = mxCalloc(eNnrn, sizeof(mwSize));
	Neicon122 = mxCalloc(eNnrn, sizeof(mwSize));


	/* Allocating matrices:*/
	eSPC = mxCalloc(totNnrnt * eCycSize, sizeof(double));
	/* array with the number of excitatory spikes arriving on each neuron from the network*/
	/*The first totNnrn elements represents the excitatory spike arriving on each neurons at the first time step of the cycle*/
	iSPC = mxCalloc(totNnrnt * iCycSize, sizeof(double));
	eSPC2 = mxCalloc(totNnrnt * eCycSize2, sizeof(double));
	iSPC2 = mxCalloc(totNnrnt * iCycSize2, sizeof(double));

	eSPC3 = mxCalloc(totNnrnt * eCycSize3, sizeof(double));
	iSPC3 = mxCalloc(totNnrnt * iCycSize3, sizeof(double));

	eSPC122 = mxCalloc(eNnrn * eCycSizeinter, sizeof(double));
	eiSPC122 = mxCalloc(iNnrn * eCycSizeinter, sizeof(double));


	A = mxCalloc(totNnrnt * totNnrnt, sizeof(mwSize));
	/* The first totNnrn elements are the connections of the neuron 1 and so on*/

	A2 = mxCalloc(totNnrnt * totNnrnt, sizeof(mwSize));
	A3 = mxCalloc(totNnrnt * totNnrnt, sizeof(mwSize));


	A122 = mxCalloc(eNnrn * eNnrn, sizeof(mwSize));

	Aei122 = mxCalloc(eNnrn * iNnrn, sizeof(mwSize));



	/* Generating connections --------------------------------------------------------*/

	RAND_MAX_double = (double)RAND_MAX;

	// Area 1

	srand(seed1);

	for (nrn1 = 0; nrn1 < totNnrnt; nrn1++) {

		// Initializing time of last spike to -1:
		tLastSP[nrn1] = -simulLen;

		for (nrn2 = 0; nrn2 < totNnrnt; nrn2++) {
			if (nrn2 != nrn1){
				rndNum = ((double)rand()) / RAND_MAX_double; /* 0 <= rndNum <= 1 */

				if (rndNum < p) {
					A[Ncon[nrn1] + nrn1*totNnrnt] = nrn2;
					Ncon[nrn1]++;
				}
			}
		}
	}

	// Area 2

	srand(seed3);

	for (nrn1 = 0; nrn1 < totNnrnt; nrn1++) {

		tLastSP[nrn1 + totNnrnt] = -simulLen;

		for (nrn2 = 0; nrn2 < totNnrnt; nrn2++) {
			if (nrn2 != nrn1){
				rndNum = ((double)rand()) / RAND_MAX_double; /* 0 <= rndNum <= 1 */

				if (rndNum < p) {
					A2[Ncon2[nrn1] + nrn1*totNnrnt] = nrn2;
					Ncon2[nrn1]++;
				}
			}
		}
	}

	// Area 3

	srand(seed5);

	for (nrn1 = 0; nrn1 < totNnrnt; nrn1++) {

		tLastSP[nrn1 + (totNnrnt * 2)] = -simulLen;

		for (nrn2 = 0; nrn2 < totNnrnt; nrn2++) {
			if (nrn2 != nrn1){
				rndNum = ((double)rand()) / RAND_MAX_double; /* 0 <= rndNum <= 1 */

				if (rndNum < p) {
					A3[Ncon3[nrn1] + nrn1*totNnrnt] = nrn2;
					Ncon3[nrn1]++;
				}
			}
		}
	}

	// Interareal

	srand(seed4);
	for (nrn1 = 0; nrn1 < eNnrn; nrn1++) {

		for (nrn2 = 0; nrn2 < eNnrn; nrn2++) {
			rndNum = ((double)rand()) / RAND_MAX_double; /* 0 <= rndNum <= 1 */

			if (rndNum < p) {
				A122[Ncon122[nrn1] + nrn1*eNnrn] = nrn2;
				Ncon122[nrn1]++;
			}
		}
	}

	for (nrn1 = 0; nrn1 < eNnrn; nrn1++) {

		for (nrn2 = 0; nrn2 < iNnrn; nrn2++) {
			rndNum = ((double)rand()) / RAND_MAX_double; /* 0 <= rndNum <= 1 */

			if (rndNum < p) {
				Aei122[Neicon122[nrn1] + nrn1*iNnrn] = nrn2;
				Neicon122[nrn1]++;
			}
		}
	}

	/* Initializing random seed for Poisson processes:*/
	srand(seed2);

	//initializing counters of spikes for firing rates 
	eCounter = 0;
	iCounter = 0;
	eCounter2 = 0;
	iCounter2 = 0;
	eCounter3 = 0;
	iCounter3 = 0;

	printf("\nSimulation started.\n");
	/* Loop over time -----------------------------------------------------*/
	for (t = 0; t < simulLen; t++) {

		exp_xFR = exp(-xFR[t]);
		exp_xFR2 = exp(-xFR2[t]);
		exp_xFR3 = exp(-xFR3[t]);


		eCycIndx = (t % eCycSize)*totNnrnt;
		iCycIndx = (t % iCycSize)*totNnrnt;

		eCycIndx2 = (t % eCycSize2)*totNnrnt;
		iCycIndx2 = (t % iCycSize2)*totNnrnt;

		eCycIndx3 = (t % eCycSize3)*totNnrnt;
		iCycIndx3 = (t % iCycSize3)*totNnrnt;

		eCycIndxinter = (t % eCycSizeinter)*eNnrn;
		iCycIndxinter = (t % eCycSizeinter)*iNnrn;

		eCycSP = ((t + (mwSize)eTl) % eCycSize)*totNnrnt;
		iCycSP = ((t + (mwSize)iTl) % iCycSize)*totNnrnt;
		eCycSP2 = ((t + (mwSize)eTl2) % eCycSize2)*totNnrnt;
		iCycSP2 = ((t + (mwSize)iTl2) % iCycSize2)*totNnrnt;
		eCycSP3 = ((t + (mwSize)eTl3) % eCycSize3)*totNnrnt;
		iCycSP3 = ((t + (mwSize)iTl3) % iCycSize3)*totNnrnt;

		eCycSPinter = ((t + (mwSize)eTlinter) % eCycSizeinter)*eNnrn;
		iCycSPinter = ((t + (mwSize)eTlinter) % eCycSizeinter)*iNnrn;

		/* Loop over neurons ----------------------------------------------*/
		for (nrn = 0; nrn < totNnrn; nrn++) {

			aI0 = aI[nrn];
			xI0 = xI[nrn];
			gI0 = gI[nrn];
			iI0 = iI[nrn];
			uI0 = uI[nrn];

			synI = xI0 + aI0 - gI0 + iI0 + uI0;

			if (nrn < eNnrn){ //excitatory neurons area 1

				//generate external Poisson input
				xSPC = -1;
				p = 1.0;
				do {
					xSPC += 1.0;
					rndNum = ((double)rand()) / RAND_MAX_double;
					p *= rndNum;
				} while (p > exp_xFR);


				e2eI[t] += aI0;

				e2eSPC = eSPC[nrn + eCycIndx];//spikes contributing to the current
				C = eTm * (e2eJ * e2eSPC);//current

				///integration e2ecurrent (following Eqs.2 3 Mazzoni 2008 but only the internal ampa current)
				aX0 = aX[nrn];

				aX_k1 = ((C - aX0) / e2eTr) / 2;
				aI_k1 = ((aX0 - aI0) / e2eTd) / 2;

				aX_k2 = ((C - aX0 - aX_k1) / e2eTr);
				aI_k2 = ((aX0 - aI0 + aX_k1 - aI_k1) / e2eTd);

				aX[nrn] += aX_k2;
				aI[nrn] += aI_k2;

				////////////////////


				x2eI[t] += xI0;
				C = eTm * (x2eJ * xSPC);//external current

				// integration external ampa current (Eqs 2 and 3 mazzoni08)
				xX0 = xX[nrn];

				xX_k1 = ((C - xX0) / e2eTr) / 2;
				xI_k1 = ((xX0 - xI0) / e2eTd) / 2;


				xX_k2 = ((C - xX0 - xX_k1) / e2eTr);
				xI_k2 = ((xX0 - xI0 + xX_k1 - xI_k1) / e2eTd);

				xX[nrn] += xX_k2;
				xI[nrn] += xI_k2;

				//////////////////////////////////

				i2eI[t] += gI0; /*//currents GABA on excitatory neurons*/


				i2eSPC = iSPC[nrn + iCycIndx];

				C = eTm * i2eJ * i2eSPC;///inhibitory current

				//integration inhibitory current (Eqs 4 and 5 Mazzoni)
				gX0 = gX[nrn];

				gX_k1 = ((C - gX0) / iTr) / 2;
				gI_k1 = ((gX0 - gI0) / iTd) / 2;


				gX_k2 = ((C - gX0 - gX_k1) / iTr);
				gI_k2 = ((gX0 - gI0 + gX_k1 - gI_k1) / iTd);

				gX[nrn] += gX_k2;
				gI[nrn] += gI_k2;


				///////////////////////////////////

				synI_k1 = xI_k1 + aI_k1 - gI_k1 + iI_k1;//total current

				V0 = V[nrn];
				Ve[t] += V0;

				if (t - tLastSP[nrn] >= eTrp) {    ///if out of the refractory period
					///integration of the voltage (Eq 1 Mazzoni 08)      
					V_k1 = ((synI - V0) / eTm) / 2;
					V_k2 = ((synI - V0 + synI_k1 - V_k1) / eTm);

					V[nrn] += V_k2;

				}

				//////////////////////////////////////////

				if (V[nrn] > Vthr) { //spike generation

					tLastSP[nrn] = t; //update time last spike for refractory period

					V[nrn] = eVres;
					eCounter++;

					for (con = 0; con < Ncon[nrn]; con++) {
						a = A[con + nrn*totNnrnt];/*in A you have conexions from nrn to others*/
						eSPC[a + eCycSP]++;//update number of spikes arriving to neurons to which nrn is connectes
					}

					for (con = 0; con < Ncon122[nrn]; con++) {
						a = A122[con + nrn*eNnrn];/*you have conexions from nrn to others*/
						eSPC122[a + eCycSPinter]++;
					}
					for (con = 0; con < Neicon122[nrn]; con++) {
						a = Aei122[con + nrn*iNnrn];/*you have conexions from nrn to others*/
						eiSPC122[a + iCycSPinter]++;
					}
				}
				eSPC[nrn + eCycIndx] = 0; //empty the actual time in the cycle
				iSPC[nrn + iCycIndx] = 0;
			}
			else if (nrn < totNnrnt) { ///inhibitory neurons area 1, the same is done but for the corresponding currents

				//generate external Poisson input
				xSPC = -1;
				p = 1.0;
				do {
					xSPC += 1.0;
					rndNum = ((double)rand()) / RAND_MAX_double;
					p *= rndNum;
				} while (p > exp_xFR);

				/////////////////////////////////////////

				//e2iI[t] += aI0;

				///integration e2icurrent (following Eqs.2 3 Mazzoni 2008 but only the internal ampa current)
				e2iSPC = eSPC[nrn + eCycIndx];

				C = iTm * (e2iJ * e2iSPC);

				aX0 = aX[nrn];

				aX_k1 = ((C - aX0) / e2iTr) / 2;
				aI_k1 = ((aX0 - aI0) / e2iTd) / 2;


				aX_k2 = ((C - aX0 - aX_k1) / e2iTr);
				aI_k2 = ((aX0 - aI0 + aX_k1 - aI_k1) / e2iTd);

				aX[nrn] += aX_k2;
				aI[nrn] += aI_k2;

				//////////////////////////////////

				//x2iI[t] += xI0;
				//integrate external current to inhibitory
				C = iTm * (x2iJ * xSPC);

				xX0 = xX[nrn];

				xX_k1 = ((C - xX0) / e2iTr) / 2;
				xI_k1 = ((xX0 - xI0) / e2iTd) / 2;


				xX_k2 = ((C - xX0 - xX_k1) / e2iTr);
				xI_k2 = ((xX0 - xI0 + xX_k1 - xI_k1) / e2iTd);

				xX[nrn] += xX_k2;
				xI[nrn] += xI_k2;

				/////////////////////////////////

				//i2iI[t] += gI0;

				///integrate inhibitory -inhibitory current
				i2iSPC = iSPC[nrn + iCycIndx];

				C = iTm * i2iJ * i2iSPC;

				gX0 = gX[nrn];

				gX_k1 = ((C - gX0) / iTr) / 2;
				gI_k1 = ((gX0 - gI0) / iTd) / 2;


				gX_k2 = ((C - gX0 - gX_k1) / iTr);
				gI_k2 = ((gX0 - gI0 + gX_k1 - gI_k1) / iTd);

				gX[nrn] += gX_k2;
				gI[nrn] += gI_k2;

				///////////////////////////////////////

				synI_k1 = xI_k1 + aI_k1 - gI_k1 + uI_k1;

				V0 = V[nrn];

				Vi[t] += V0;

				if (t - tLastSP[nrn] >= iTrp) {

					V_k1 = ((synI - V0) / iTm) / 2;
					V_k2 = ((synI - V0 + synI_k1 - V_k1) / iTm);

					V[nrn] += V_k2;
				}

				/////////////////////////////////////

				if (V[nrn] > Vthr) {

					tLastSP[nrn] = t;

					V[nrn] = iVres;
					iCounter++;

					for (con = 0; con < Ncon[nrn]; con++) {
						a = A[con + nrn*totNnrnt];/*in A you have conexions from nrn to others*/
						iSPC[a + iCycSP]++;
					}

				}
				eSPC[nrn + eCycIndx] = 0;
				iSPC[nrn + iCycIndx] = 0;
			}
			else if (nrn < totNnrnt + eNnrn) {///excitatory neurons area 2 (all is analogous to excitatory of area 1)

				xSPC2 = -1;
				p = 1.0;
				do {
					xSPC2 += 1.0;
					rndNum = ((double)rand()) / RAND_MAX_double;
					p *= rndNum;
				} while (p > exp_xFR2);

				///////////////////////////////////////

				e2eI2[t] += aI0;


				e2eSPC2 = eSPC2[nrn - totNnrnt + eCycIndx2];

				C = eTm2 * (e2eJ2 * e2eSPC2);

				aX0 = aX[nrn];

				aX_k1 = ((C - aX0) / e2eTr2) / 2;
				aI_k1 = ((aX0 - aI0) / e2eTd2) / 2;


				aX_k2 = ((C - aX0 - aX_k1) / e2eTr2);
				aI_k2 = ((aX0 - aI0 + aX_k1 - aI_k1) / e2eTd2);

				aX[nrn] += aX_k2;
				aI[nrn] += aI_k2;

				////////////////////////////////////


				x2eI2[t] += xI0;

				C = eTm2 * (x2eJ2 * xSPC2);

				xX0 = xX[nrn];

				xX_k1 = ((C - xX0) / e2eTr2) / 2;
				xI_k1 = ((xX0 - xI0) / e2eTd2) / 2;


				xX_k2 = ((C - xX0 - xX_k1) / e2eTr2);
				xI_k2 = ((xX0 - xI0 + xX_k1 - xI_k1) / e2eTd2);

				xX[nrn] += xX_k2;
				xI[nrn] += xI_k2;

				////////////////////////////////////

				i2eI2[t] += gI0;


				i2eSPC2 = iSPC2[nrn - totNnrnt + iCycIndx2];

				C = eTm2 * i2eJ2 * i2eSPC2;

				gX0 = gX[nrn];

				gX_k1 = ((C - gX0) / iTr2) / 2;
				gI_k1 = ((gX0 - gI0) / iTd2) / 2;


				gX_k2 = ((C - gX0 - gX_k1) / iTr2);
				gI_k2 = ((gX0 - gI0 + gX_k1 - gI_k1) / iTd2);

				gX[nrn] += gX_k2;
				gI[nrn] += gI_k2;

				////////////////////////////////////

				e2eI122[t] += iI0;


				e2eSPC122 = eSPC122[nrn - totNnrnt + eCycIndxinter];

				C = eTm2 * (e2eJ122 * e2eSPC122);

				iX0 = iX[nrn];

				iX_k1 = ((C - iX0) / e2eTr2) / 2;
				iI_k1 = ((iX0 - iI0) / e2eTd2) / 2;


				iX_k2 = ((C - iX0 - iX_k1) / e2eTr2);
				iI_k2 = ((iX0 - iI0 + iX_k1 - iI_k1) / e2eTd2);

				iX[nrn] += iX_k2;
				iI[nrn] += iI_k2;

				//////////////////////////////////////

				synI_k1 = xI_k1 + aI_k1 - gI_k1 + iI_k1;

				V0 = V[nrn];
				Ve2[t] += V0;

				if (t - tLastSP[nrn] >= eTrp2) {

					V_k1 = ((synI - V0) / eTm2) / 2;
					V_k2 = ((synI - V0 + synI_k1 - V_k1) / eTm2);

					V[nrn] += V_k2;
				}

				/////////////////////////////////////////

				if (V[nrn] > Vthr) {

					tLastSP[nrn] = t;


					V[nrn] = eVres;
					eCounter2++;

					for (con = 0; con < Ncon2[nrn - totNnrnt]; con++) {
						a = A2[con + (nrn - totNnrnt)*totNnrnt];/*in A you have conexions from nrn to others*/
						eSPC2[a + eCycSP2]++;
					}
				}
				eSPC2[nrn - totNnrnt + eCycIndx2] = 0;
				iSPC2[nrn - totNnrnt + iCycIndx2] = 0;
				eSPC122[nrn - totNnrnt + eCycIndxinter] = 0;
			}
			else if (nrn < totNnrnt * 2) { //inhibitory of area 2 ()

				xSPC2 = -1;
				p = 1.0;
				do {
					xSPC2 += 1.0;
					rndNum = ((double)rand()) / RAND_MAX_double;
					p *= rndNum;
				} while (p > exp_xFR2);

				////////////////////////////////////////////

				//e2iI2[t] += aI0;

				e2iSPC2 = eSPC2[nrn - totNnrnt + eCycIndx2];

				C = iTm2 * (e2iJ2 * e2iSPC2);

				aX0 = aX[nrn];

				aX_k1 = ((C - aX0) / e2iTr2) / 2;
				aI_k1 = ((aX0 - aI0) / e2iTd2) / 2;


				aX_k2 = ((C - aX0 - aX_k1) / e2iTr2);
				aI_k2 = ((aX0 - aI0 + aX_k1 - aI_k1) / e2iTd2);

				aX[nrn] += aX_k2;
				aI[nrn] += aI_k2;

				////////////////////////////////////


				//x2iI2[t] += xI0;

				C = iTm2 * (x2iJ2 * xSPC2);

				xX0 = xX[nrn];

				xX_k1 = ((C - xX0) / e2iTr2) / 2;
				xI_k1 = ((xX0 - xI0) / e2iTd2) / 2;


				xX_k2 = ((C - xX0 - xX_k1) / e2iTr2);
				xI_k2 = ((xX0 - xI0 + xX_k1 - xI_k1) / e2iTd2);

				xX[nrn] += xX_k2;
				xI[nrn] += xI_k2;

				///////////////////////////////////////

				//i2iI2[t] += gI0;


				i2iSPC2 = iSPC2[nrn - totNnrnt + iCycIndx2];

				C = iTm2 * i2iJ2 * i2iSPC2;

				gX0 = gX[nrn];

				gX_k1 = ((C - gX0) / iTr2) / 2;
				gI_k1 = ((gX0 - gI0) / iTd2) / 2;


				gX_k2 = ((C - gX0 - gX_k1) / iTr2);
				gI_k2 = ((gX0 - gI0 + gX_k1 - gI_k1) / iTd2);

				gX[nrn] += gX_k2;
				gI[nrn] += gI_k2;

				//////////////////////////////////////////

				e2iI122[t] += uI0;


				e2iSPC122 = eiSPC122[nrn - totNnrnt - eNnrn + iCycIndxinter];

				C = iTm2 * (e2iJ122 * e2iSPC122);

				uX0 = uX[nrn];

				uX_k1 = ((C - uX0) / e2iTr2) / 2;
				uI_k1 = ((uX0 - uI0) / e2iTd2) / 2;


				uX_k2 = ((C - uX0 - uX_k1) / e2iTr2);
				uI_k2 = ((uX0 - uI0 + uX_k1 - uI_k1) / e2iTd2);

				uX[nrn] += uX_k2;
				uI[nrn] += uI_k2;

				//////////////////////////////////////


				synI_k1 = xI_k1 + aI_k1 - gI_k1 + uI_k1;

				V0 = V[nrn];
				Vi2[t] += V0;

				if (t - tLastSP[nrn] >= iTrp2) {

					V_k1 = ((synI - V0) / iTm2) / 2;
					V_k2 = ((synI - V0 + synI_k1 - V_k1) / iTm2);

					V[nrn] += V_k2;
				}

				/////////////////////////////////////////

				if (V[nrn] > Vthr) {

					tLastSP[nrn] = t;


					V[nrn] = iVres;
					iCounter2++;

					for (con = 0; con < Ncon2[nrn - totNnrnt]; con++) {
						a = A2[con + (nrn - totNnrnt)*totNnrnt];/*in A you have conexions from nrn to others*/
						iSPC2[a + iCycSP2]++;
					}

				}
				eSPC2[nrn - totNnrnt + eCycIndx2] = 0;
				iSPC2[nrn - totNnrnt + iCycIndx2] = 0;
				eiSPC122[nrn - totNnrnt - eNnrn + iCycIndxinter] = 0;
			}
			else if (nrn < (totNnrnt * 2) + eNnrn) { // excitatory area 3
				xSPC3 = -1;
				p = 1.0;
				do {
					xSPC3 += 1.0;
					rndNum = ((double)rand()) / RAND_MAX_double;
					p *= rndNum;
				} while (p > exp_xFR3);

				///////////////////////////////////////

				e2eI3[t] += aI0;


				e2eSPC3 = eSPC3[nrn - (totNnrnt * 2) + eCycIndx3];

				C = eTm3 * (e2eJ3 * e2eSPC3);

				aX0 = aX[nrn];

				aX_k1 = ((C - aX0) / e2eTr3) / 2;
				aI_k1 = ((aX0 - aI0) / e2eTd3) / 2;


				aX_k2 = ((C - aX0 - aX_k1) / e2eTr3);
				aI_k2 = ((aX0 - aI0 + aX_k1 - aI_k1) / e2eTd3);

				aX[nrn] += aX_k2;
				aI[nrn] += aI_k2;

				////////////////////////////////////


				x2eI3[t] += xI0;

				C = eTm3 * (x2eJ3 * xSPC3);

				xX0 = xX[nrn];

				xX_k1 = ((C - xX0) / e2eTr3) / 2;
				xI_k1 = ((xX0 - xI0) / e2eTd3) / 2;


				xX_k2 = ((C - xX0 - xX_k1) / e2eTr3);
				xI_k2 = ((xX0 - xI0 + xX_k1 - xI_k1) / e2eTd3);

				xX[nrn] += xX_k2;
				xI[nrn] += xI_k2;

				////////////////////////////////////

				i2eI3[t] += gI0;


				i2eSPC3 = iSPC3[nrn - (totNnrnt * 2) + iCycIndx3];

				C = eTm3 * i2eJ3 * i2eSPC3;

				gX0 = gX[nrn];

				gX_k1 = ((C - gX0) / iTr3) / 2;
				gI_k1 = ((gX0 - gI0) / iTd3) / 2;


				gX_k2 = ((C - gX0 - gX_k1) / iTr3);
				gI_k2 = ((gX0 - gI0 + gX_k1 - gI_k1) / iTd3);

				gX[nrn] += gX_k2;
				gI[nrn] += gI_k2;

				//////////////////////////////////////

				synI_k1 = xI_k1 + aI_k1 - gI_k1 + iI_k1;

				V0 = V[nrn];
				Ve3[t] += V0;

				if (t - tLastSP[nrn] >= eTrp3) {

					V_k1 = ((synI - V0) / eTm3) / 2;
					V_k2 = ((synI - V0 + synI_k1 - V_k1) / eTm3);

					V[nrn] += V_k2;
				}

				/////////////////////////////////////////

				if (V[nrn] > Vthr) {

					tLastSP[nrn] = t;


					V[nrn] = eVres;
					eCounter3++;

					for (con = 0; con < Ncon3[nrn - (totNnrnt * 2)]; con++) {
						a = A3[con + (nrn - (totNnrnt * 2))*totNnrnt];/*in A you have conexions from nrn to others*/
						eSPC3[a + eCycSP3]++;
					}
				}
				eSPC3[nrn - (totNnrnt * 2) + eCycIndx3] = 0;
				iSPC3[nrn - (totNnrnt * 2) + iCycIndx3] = 0;

			}
			else { // inhibitory area 3
				xSPC3 = -1;
				p = 1.0;
				do {
					xSPC3 += 1.0;
					rndNum = ((double)rand()) / RAND_MAX_double;
					p *= rndNum;
				} while (p > exp_xFR3);

				////////////////////////////////////////////

				//e2iI2[t] += aI0;

				e2iSPC3 = eSPC3[nrn - (totNnrnt * 2) + eCycIndx3];

				C = iTm3 * (e2iJ3 * e2iSPC3);

				aX0 = aX[nrn];

				aX_k1 = ((C - aX0) / e2iTr3) / 2;
				aI_k1 = ((aX0 - aI0) / e2iTd3) / 2;


				aX_k2 = ((C - aX0 - aX_k1) / e2iTr3);
				aI_k2 = ((aX0 - aI0 + aX_k1 - aI_k1) / e2iTd3);

				aX[nrn] += aX_k2;
				aI[nrn] += aI_k2;

				////////////////////////////////////


				//x2iI2[t] += xI0;

				C = iTm3 * (x2iJ3 * xSPC3);

				xX0 = xX[nrn];

				xX_k1 = ((C - xX0) / e2iTr3) / 2;
				xI_k1 = ((xX0 - xI0) / e2iTd3) / 2;


				xX_k2 = ((C - xX0 - xX_k1) / e2iTr3);
				xI_k2 = ((xX0 - xI0 + xX_k1 - xI_k1) / e2iTd3);

				xX[nrn] += xX_k2;
				xI[nrn] += xI_k2;

				///////////////////////////////////////

				//i2iI2[t] += gI0;


				i2iSPC3 = iSPC3[nrn - (totNnrnt * 2) + iCycIndx3];

				C = iTm3 * i2iJ3 * i2iSPC3;

				gX0 = gX[nrn];

				gX_k1 = ((C - gX0) / iTr3) / 2;
				gI_k1 = ((gX0 - gI0) / iTd3) / 2;


				gX_k2 = ((C - gX0 - gX_k1) / iTr3);
				gI_k2 = ((gX0 - gI0 + gX_k1 - gI_k1) / iTd3);

				gX[nrn] += gX_k2;
				gI[nrn] += gI_k2;

				//////////////////////////////////////


				synI_k1 = xI_k1 + aI_k1 - gI_k1 + uI_k1;

				V0 = V[nrn];
				Vi3[t] += V0;

				if (t - tLastSP[nrn] >= iTrp3) {

					V_k1 = ((synI - V0) / iTm3) / 2;
					V_k2 = ((synI - V0 + synI_k1 - V_k1) / iTm3);

					V[nrn] += V_k2;
				}

				/////////////////////////////////////////

				if (V[nrn] > Vthr) {

					tLastSP[nrn] = t;


					V[nrn] = iVres;
					iCounter3++;

					for (con = 0; con < Ncon3[nrn - (totNnrnt * 2)]; con++) {
						a = A3[con + (nrn - (totNnrnt * 2))*totNnrnt];/*in A you have conexions from nrn to others*/
						iSPC3[a + iCycSP3]++;
					}

				}
				eSPC3[nrn - (totNnrnt * 2) + eCycIndx3] = 0;
				iSPC3[nrn - (totNnrnt * 2) + iCycIndx3] = 0;
			}


		} /* end of loop over neurons*/

		///time series of firing rates
		eFR[t + 1] = eCounter;
		eCounter = 0;
		iFR[t + 1] = iCounter;
		iCounter = 0;

		eFR2[t + 1] = eCounter2;
		eCounter2 = 0;
		iFR2[t + 1] = iCounter2;
		iCounter2 = 0;

		eFR3[t + 1] = eCounter3;
		eCounter3 = 0;
		iFR3[t + 1] = iCounter3;
		iCounter3 = 0;
	} /* end of loop over time*/

	/// Free memory for the variables
	mxFree(tLastSP);

	mxFree(aX);
	mxFree(gX);
	mxFree(xX);
	mxFree(iX);
	mxFree(uX);

	mxFree(aI);
	mxFree(gI);
	mxFree(xI);
	mxFree(iI);
	mxFree(uI);

	mxFree(V);
	mxFree(Ncon);
	mxFree(Ncon2);
	mxFree(Ncon3);
	mxFree(Ncon122);
	mxFree(Neicon122);


	mxFree(eSPC);
	mxFree(iSPC);
	mxFree(eSPC2);
	mxFree(iSPC2);
	mxFree(eSPC3);
	mxFree(iSPC3);
	mxFree(eSPC122);
	mxFree(eiSPC122);


	mxFree(A);
	mxFree(A2);
	mxFree(A3);
	mxFree(A122);
	mxFree(Aei122);


	printf("Simulation done.\n\n");

} /* main end*/