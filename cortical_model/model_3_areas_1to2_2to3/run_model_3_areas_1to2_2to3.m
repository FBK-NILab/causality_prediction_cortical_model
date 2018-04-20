
clear;

rand('state',sum(100*clock));

%%set number of neurons
eNnrn = 4000; iNnrn = 1000;

%%%values of time constants are fixed and are the same for both areas and
%%%the same as in Mazzoni 08.
%%%set time constants area 1
eTm = 20; iTm = 10;
eTl = 1;  iTl = 1;
e2eTr = 0.4; e2eTd = 2;
e2iTr = 0.2; e2iTd = 1;
iTr = 0.25;  iTd = 5;

%%%set time constant area 2
eTm2 = 20; iTm2 = 10;
eTl2 = 1;  iTl2 = 1;
e2eTr2 = 0.4; e2eTd2 = 2;
e2iTr2 = 0.2; e2iTd2 = 1;
iTr2 = 0.25;  iTd2 = 5;

%%%set time constant area 2
eTm3 = 20; iTm3 = 10;
eTl3 = 1;  iTl3 = 1;
e2eTr3 = 0.4; e2eTd3 = 2;
e2iTr3 = 0.2; e2iTd3 = 1;
iTr3 = 0.25;  iTd3 = 5;

%%set inter areal time constant
eTlinter = 3;

%%%synaptic efficacies area 1
%%% here e2e are changed with respect to Mazzoni08 to compensate
%%% that there are the interareal currents

i2i = 2.7;
e2e = 0.37;
e2i = 0.7;
i2e = 1.7;
x2i = 0.95;
x2e = 0.55;

%%%synaptic efficacies area 2
i2i2 = 2.7;
e2e2 = 0.37;
e2i2 = 0.7;
i2e2 = 1.7;
x2i2 = 0.95;
x2e2 = 0.55;

%%%synaptic efficacies area 3
i2i3 = 2.7;
e2e3 = 0.37;
e2i3 = 0.7;
i2e3 = 1.7;
x2i3 = 0.95;
x2e3 = 0.55;

% e2e122s = [0.15 0.25 ];
% e2e221s = [0.15 0.25 ];

% e2e122 = 0.25;
% e2i122 = e2e122;
% 
% e2e322 = 0.25;
% e2i322 = e2e322;

Dt = 0.1; %integration step in ms
ts = 6000; %length time-series in ms
simulLen = ts/Dt;

%%%seeds for random generation
seed1=3; %% Area 1 connectivity
seed2=1;
seed3 = 5; %% Area 2 connectivity
seed4 = 6;
seed5 = 2; %% Area 3 connectivity

numberOfSamples = 6000;

output1 = zeros(numberOfSamples, ts);
output2 = zeros(numberOfSamples, ts);
output3 = zeros(numberOfSamples, ts);

numWorkers = 40;

if (isempty(gcp('nocreate')))
    parpool(numWorkers);
else
    poolTemp = gcp;
    if(poolTemp.NumWorkers < numWorkers)
        delete(gcp)
        parpool(numWorkers);
    end
end

tic

rng(5000, 'twister');   
synpaticEfficacies = rand(numberOfSamples, 2) .* 0.18;


parfor n = 1:numberOfSamples 

    e2e322 = synpaticEfficacies(n, 1);
    e2i322 = e2e322;
    
    e2e123 = synpaticEfficacies(n, 2);
    e2i123 = e2e123;
    
    %%%%generate the constant component of the input external rate
    signal1 = 2; %constant external input rate spikes/ms to area 1
    signal2 = 2; %same to area 2
    signal3 = 2; %same to area 3
    input = ones(simulLen,1) * signal1 * Dt;
    input2 = ones(simulLen,1) * signal2 * Dt;
    input3 = ones(simulLen,1) * signal3 * Dt;

    %%%generate the noisy component of the input rate
    tau=16; %in ms
    sigma=0.4; %in sqrt(spk/ms)
    seeds = [n, n + numberOfSamples, n + numberOfSamples * 2];
    seeds = seeds(randperm(3));
    
    rseed = seeds(1);
    rseed2 = seeds(2);
    rseed3 = seeds(3);
    
     %%% the noise are Ornstein–Uhlenbeck processes (power spectrum almost white for low frequencies and fast decay for high frequencies)
    noiseFR = OU_euler_seed(simulLen, Dt, tau, sigma,rseed)*Dt;
    noiseFR2 = OU_euler_seed(simulLen, Dt, tau, sigma,rseed2)*Dt;
    noiseFR3 = OU_euler_seed(simulLen, Dt, tau, sigma,rseed3)*Dt;

    %%%total constant plus noise input
    input = input + noiseFR;
    input2 = input2 + noiseFR2;
    input3 = input3 + noiseFR3;
    %%%no negative rates
    input(input<0) = 0;
    input2(input2<0) = 0;
    input3(input3<0) = 0;

    %%%initialize time series.
    %%% Here it is done to explore a set of two parameters. This parameters can
    %%% be the one you decide but the name of the variables are if you were
    %%% exploring a set of e2e221 synaptic efficacies and a set of e2e122 synaptic efficacies 

    %%%%area 1
    eFRs = zeros(1,ts);%%excitatory firing rate
    iFRs = zeros(1,ts);%%inhibitory firing rate
    e2eIs = zeros(1,ts);%%excitatory-excitatory current
    i2eIs = zeros(1,ts);%%inhibitory-excitatory current
    % e2iIs = zeros(1,ts);
    % i2iIs = zeros(1,ts);
    x2eIs = zeros(1,ts);%%external-excitatory current
    % x2iIs = zeros(1,ts);
    Ves = zeros(1,ts);%%voltage excitatory neurons
    Vis = zeros(1,ts);%%voltage inhibitory neurons

    %%%same for area 2
    eFRs2 = zeros(1,ts);
    iFRs2 = zeros(1,ts);
    e2eIs2 = zeros(1,ts);
    i2eIs2 = zeros(1,ts);
    % e2iIs2 = zeros(1,ts);
    % i2iIs2 = zeros(1,ts);
    x2eIs2 = zeros(1,ts);
    % x2iIs2 = zeros(1,ts);
    Ves2 = zeros(1,ts);
    Vis2 = zeros(1,ts);

    %%%same for area 3
    eFRs3 = zeros(1,ts);
    iFRs3 = zeros(1,ts);
    e2eIs3 = zeros(1,ts);
    i2eIs3 = zeros(1,ts);
    % e2iIs3 = zeros(1,ts);
    % i2iIs3 = zeros(1,ts);
    x2eIs3 = zeros(1,ts);
    % x2iIs3 = zeros(1,ts);
    Ves3 = zeros(1,ts);
    Vis3 = zeros(1,ts);

    %%%interareal currents
    e2eI322 = zeros(1,ts);
    e2iI322 = zeros(1,ts);

    e2eI123 = zeros(1,ts);
    e2iI123 = zeros(1,ts);
    
    dt2 = Dt;


    tic
    [e2eIs0, i2eIs0, eFRs0, iFRs0, x2eIs0, Ves0, Vis0, e2eIs20, i2eIs20, eFRs20, iFRs20, x2eIs20, Ves20, Vis20, e2eIs30, i2eIs30, eFRs30, iFRs30, x2eIs30, Ves30, Vis30, e2eI3220, e2iI3220, e2eI1230, e2iI1230] = model_3_areas_1to2_2to3(Dt, input, input2, input3, seed1, seed2, seed3, seed4, seed5, eTm, iTm, eTl, iTl, e2eTr, e2eTd, e2iTr, e2iTd,iTr, iTd, eTm2, iTm2, eTl2, iTl2, e2eTr2, e2eTd2, e2iTr2, e2iTd2,iTr2, iTd2, eTm3, iTm3, eTl3, iTl3, e2eTr3, e2eTd3, e2iTr3, e2iTd3,iTr3, iTd3, eTlinter, i2i,e2i,x2i,i2e,e2e,x2e, i2i2,e2i2,x2i2,i2e2,e2e2,x2e2, i2i3,e2i3,x2i3,i2e3,e2e3,x2e3,e2e322,e2i322,e2e123,e2i123,eNnrn, iNnrn);
    toc


    %%%undersampling, the output of the function is sampled at 1ms/Dt
    %%%and here we change to 1ms
    for i = 1:ts

        eFRs(i)= sum(eFRs0(1+(i-1)/dt2:i/dt2));
        iFRs(i)= sum(iFRs0(1+(i-1)/dt2:i/dt2));
        e2eIs(i)= sum(e2eIs0(1+(i-1)/dt2:i/dt2));
        i2eIs(i)= sum(i2eIs0(1+(i-1)/dt2:i/dt2));
        %e2iIs(i)= sum(e2iIs0(1+(i-1)/dt2:i/dt2));
        %i2iIs(i)= sum(i2iIs0(1+(i-1)/dt2:i/dt2));
        x2eIs(i)= sum(x2eIs0(1+(i-1)/dt2:i/dt2));
        %x2iIs(i)= sum(x2iIs0(1+(i-1)/dt2:i/dt2));
        Ves(i)= sum(Ves0(1+(i-1)/dt2:i/dt2));
        Vis(i)= sum(Vis0(1+(i-1)/dt2:i/dt2));

        eFRs2(i)= sum(eFRs20(1+(i-1)/dt2:i/dt2));
        iFRs2(i)= sum(iFRs20(1+(i-1)/dt2:i/dt2));
        e2eIs2(i)= sum(e2eIs20(1+(i-1)/dt2:i/dt2));
        i2eIs2(i)= sum(i2eIs20(1+(i-1)/dt2:i/dt2));
        %e2iIs2(i)= sum(e2iIs20(1+(i-1)/dt2:i/dt2));
        %i2iIs2(i)= sum(i2iIs20(1+(i-1)/dt2:i/dt2));
        x2eIs2(i)= sum(x2eIs20(1+(i-1)/dt2:i/dt2));
        %x2iIs2(i)= sum(x2iIs20(1+(i-1)/dt2:i/dt2));
        Ves2(i)= sum(Ves20(1+(i-1)/dt2:i/dt2));
        Vis2(i)= sum(Vis20(1+(i-1)/dt2:i/dt2));

        eFRs3(i)= sum(eFRs30(1+(i-1)/dt2:i/dt2));
        iFRs3(i)= sum(iFRs30(1+(i-1)/dt2:i/dt2));
        e2eIs3(i)= sum(e2eIs30(1+(i-1)/dt2:i/dt2));
        i2eIs3(i)= sum(i2eIs30(1+(i-1)/dt2:i/dt2));
        %e2iIs3(i)= sum(e2iIs30(1+(i-1)/dt2:i/dt2));
        %i2iIs3(i)= sum(i2iIs30(1+(i-1)/dt2:i/dt2));
        x2eIs3(i)= sum(x2eIs30(1+(i-1)/dt2:i/dt2));
        %x2iIs3(i)= sum(x2iIs30(1+(i-1)/dt2:i/dt2));
        Ves3(i)= sum(Ves30(1+(i-1)/dt2:i/dt2));
        Vis3(i)= sum(Vis30(1+(i-1)/dt2:i/dt2));
        
        e2eI322(i)= sum(e2eI3220(1+(i-1)/dt2:i/dt2));
        e2iI322(i)= sum(e2iI3220(1+(i-1)/dt2:i/dt2));
        
        e2eI123(i)= sum(e2eI1230(1+(i-1)/dt2:i/dt2));
        e2iI123(i)= sum(e2iI1230(1+(i-1)/dt2:i/dt2));
    end



    %%%%%%%%%%%construct LFPs and total synaptic currents
    %%% LFP = sum|currents to excitatory cells|
    %%%area 1
    lfp = (e2eIs+ i2eIs + x2eIs) / (eNnrn/Dt); 
    %%%area 2
    lfp2 = (e2eIs2+ i2eIs2 + x2eIs2 + e2eI322) / (eNnrn/Dt);
    %%%area 3
    lfp3 = (e2eIs3+ i2eIs3 + x2eIs3 + e2eI123) / (eNnrn/Dt);
    
    output1(n, :) = lfp;
    output2(n, :) = lfp2;
    output3(n, :) = lfp3;

end

toc

output = zeros(numberOfSamples, 3, ts);
output(:,1,:) = output1;
output(:,2,:) = output2;
output(:,3,:) = output3;

% %%%total current = sum currents to excitatory cells
% syn = (e2eIs- i2eIs + x2eIs)/(eNnrn/Dt);%%notice that the inhibitory current is saved in the network as positive and the sign has to be put ad-hoc
% syn2 = (e2eIs2- i2eIs2 + x2eIs2 + e2eI122)/(eNnrn/Dt);
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save('./output/netsMultivariateConnection1to3and3to2count6000-rndMax0.18SynEff.mat', 'output', 'synpaticEfficacies');
