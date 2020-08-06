load('FakeData_All.mat')

runs = 1000;
cell_choice = 1:1:size(sps,2);
N = [1 2 3 4 5];
train_frac = 0.1; % Proportion of total time series for each training set
val_frac = 0.3; % Proportion of training set length for validation, must be less than 1

% Set params for fitting, including bases 
nkbasis = 15;  % number of basis vectors for representing k
nkt = 160;    % Number of time bins in stimulus filter k
nhbasis = 15;  % number of basis vectors for representing h
hpeakFinal = .18;   % time of peak of last basis vector for h


%% 1. CURATE DATA

slen = length(Stim);
sps = sps(1:slen,:);
Stim = Stim(1:slen);
swid = size(Stim,2);  % Stimulus width  (pixels); must match # pixels in stim filter
tinit = dt*(0:1:slen-1).'; % Index for time


%% 2. DOWNSAMPLING

down = 5; % Number of time bins combined together
slen = floor(slen/down);
nkt = round(nkt/down); 
dt = dt*down; %

tstep=tinit;
tempsps = zeros(slen,size(sps,2));
if down > 1
    tstep = mean(reshape(tstep(1:slen*down),[down slen]));
    for j=1:size(sps,2)
        tempsps(:,j) = sum(reshape(sps(1:slen*down,j),[down slen]));
    end
    Stim = interp1(tinit(1:slen*down),Stim(1:slen*down),tstep,'pchip');
end
tstep=tstep.';
Stim = Stim.';
sps = tempsps;


%% 3. SET UP TRAINING AND VALIDATION SETS

tic
for popsize = 1:length(N)
    ncells = N(popsize);
    mean_abserror_decoding = zeros(runs,1);
    stimhet = zeros(runs,1);
    posthet = zeros(runs,1);
    biashet = zeros(runs,1);
    stimulus_mat = zeros(floor(val_frac*train_frac*slen),runs);
    optimal_mat = zeros(floor(val_frac*train_frac*slen),runs);
    choleskyflag = zeros(runs,1);
    
for k=1:runs

clearvars ntrain set_index Stimtrain Stimval spstrain spsval ttrain tval slentrain slenval sta gg0 gg1 acf stimprior optStim 

% Randomly selects cells from choices WITHOUT replacement
cells = sort(datasample(cell_choice,ncells,'Replace',false));

[ntrain,set_index,Stimtrain,Stimval,spstrain,spsval,ttrain,tval,slentrain,slenval] = EncodingSets_ValFollowing(cells,train_frac,val_frac,Stim,sps,tstep);


%% 4. SET PARAMETERS AND DISPLAY FOR GLM

% ggsim = makeSimStruct_GLM(nkt,dtStim,dtSp); % Create GLM structure with default params
ggsim = makeSimStruct_GLM(nkt,dt,dt); % Create GLM structure with default params


%% 5. SETUP FITTING PARAMETERS FOR RANDOM TRAINING SET

% Compute the STA
sta = zeros(nkt,ntrain);
for j = 1:ntrain
    sta(:,j) = simpleSTC(Stimtrain,spstrain(:,j),nkt); % Compute STA
end

% Set mask (if desired)
exptmask= []; %[1 slen*dtStim];  % data range to use for fitting (in s).

for j=1:ntrain
    % gg0 = makeFittingStruct_GLM(dtStim,dtSp,nkt,nkbasis,sta,nhbasis,hpeakFinal);
    temp = makeFittingStruct_GLM(dt,dt,nkt,nkbasis,sta(:,j),nhbasis,hpeakFinal); 
    temp.sps = spstrain(:,j);
    temp.mask = exptmask; % insert mask (optional)
    temp.ihw = randn(size(temp.ihw))*1; % initialize spike-history weights randomly
    gg0(j) = temp;
end
clearvars temp
% gg.dc currently set at 0 in code. Could change to [] to allow constant to
% be calculated

negloglival0_partial = zeros(ncells,1);
rr_partial = zeros(slentrain,ncells);
for j = 1:ntrain
    % Compute conditional intensity at initial parameters 
    [tempneglog,temprr] = neglogli_GLM(gg0(j),Stimtrain);
    negloglival0_partial(j) = tempneglog;
    rr_partial(:,j) = temprr;
end
clearvars tempneglog temprr
negloglival0 = sum(negloglival0_partial);
fprintf('Initial negative log-likelihood: %.5f\n', negloglival0);


%% 6. DO ML FITTING 

% Because the cells are uncoupled, the encoding models are done separately
opts = {'display', 'iter', 'maxiter', 200}; % options for fminunc
for j = 1:ntrain
    [tempgg1, tempnegloglival] = MLfit_GLM(gg0(j),Stimtrain,opts); % do ML (requires optimization toolbox)
    tempgg1.sta = sta(:,j);
    tempgg1.RefreshRate = 1000; % Is this the proper refresh rate?
    tempgg1.dt = dt;
    gg1(j) = tempgg1;
    negloglival1_partial(j) = tempnegloglival;
end
clearvars tempgg1 tempnegloglival
negloglival1 = sum(negloglival1_partial);

fprintf('Model negative log-likelihood: %.5f\n', negloglival1);


%% 7. DETERMINE AUTOCORRELATION STRUCTURE OF STIMULUS

corrlength = floor(val_frac*slentrain);
Burn = 0;
acf = autocorr(Stimtrain(:,1),'NumLags',corrlength-1,'NumSTD',3);

acf(end+1:end+Burn) = 0;
stimCovMat = toeplitz(acf);

numStims= size(Stimval,2);


%% 8. SET UP PRIOR FOR STIMULUS DECODING

[cholMat,flag] = chol(stimCovMat); % Cholesky decomposition of cov matrix
choleskyflag(k) = flag; % Track when matrix is not invertible
dim = length(cholMat);
stimprior = zeros(dim, numStims);
% Instill autocorrelation structure on random vector
% Change from randn() to proper random number generator if non-Gaussian
for j = 1:numStims
    stimprior(:,j) = cholMat*randn(dim,1);
end

stimprior = stimprior(Burn+1:end,:);
stimCovMat= stimCovMat(Burn+1:end, Burn+1:end);


%% 9. CALL GLMs FOR REQUESTED CELLS

numModels = length(gg1);
% Rename cells called from group
cellModels=gg1;

numSimRepeats = 50;
testInterval = [0 nkt];
maxSpikes = 100;


%% 10. INITIATE BAYES STIM DECODER

stimInterval = [tval(1,:); tval(corrlength,:)];
dtDecoding = dt;
initStim=stimprior;

spikeTimes = spsval;
[optStim, exitflag, fval, hessian] = bayesStimDecoder(cellModels, spikeTimes, stimInterval, dtDecoding, stimCovMat, initStim);


%% 11. HETEROGENEITY AND ERROR CALCULATION

abserror_decoding = abs(Stimval - optStim);

abserror_decoding = abs(Stimval - optStim);
weight_uncertainty = 1./diag(hessian);
% Hessian-Weighted Error
mean_abserror_decoding(k) = (abserror_decoding.' * weight_uncertainty)/sum(weight_uncertainty);

if ncells > 1
    kMat = zeros(length(gg1(1).k),length(gg1));
    hMat = zeros(length(gg1(1).ih),length(gg1));
    dcVec = zeros(length(gg1),1);
    for m = 1:length(gg1)
        kMat(:,m) = gg1(m).k;
        hMat(:,m) = gg1(m).ih;
        dcVec(m) = gg1(m).dc;
    end

    stimdiff = 0;
    for q = 1:nkt
        stimdiff = stimdiff + mean(nonzeros(abs(kMat(q,:)-kMat(q,:)')));
    end
    stimhet(k) = stimdiff/(nkt*nchoosek(ncells,2));

    postdiff = 0;
    for q = 1:size(gg1(1).ih,1)
        postdiff = postdiff + mean(nonzeros(abs(hMat(q,:)-hMat(q,:)')));
    end
    posthet(k) = postdiff/(size(gg1(1).ih,1)*nchoosek(ncells,2));
    biashet(k) = mean(nonzeros(abs(dcVec-dcVec')));
    
    
end

    stimulus_mat(:,k) = Stimval;
    optimal_mat(:,k) = optStim;
end

Decoding_Stats(popsize).Error = mean_abserror_decoding;
Decoding_Stats(popsize).Bias_diff = biashet;
Decoding_Stats(popsize).Stim_diff = stimhet;
Decoding_Stats(popsize).Post_diff = posthet;
Decoding_Stats(popsize).Stimulus = stimulus_mat;
Decoding_Stats(popsize).Optimal = optimal_mat;
Decoding_Stats(popsize).CholeskyFlag = choleskyflag;

end
toc

save('Decoding_Stats_ValFollowing.mat','Decoding_Stats_ValFollowing')

