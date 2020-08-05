% this is the basic bayesStimDecoder, it seems to perform decoding
% correctly as long as the dt of the stimulus is 1 msec

function [optStim, exitflag, fval, hessian] = bayesStimDecoder_Fall2019(cellModels, spikeTimes, stimInterval, dtDecoding, stimCovMat, initStim)

numCells = length(cellModels);
tPts = stimInterval(1):dtDecoding:stimInterval(2);
slenval = length(tPts);
if cellModels(1).dt < dtDecoding
    nkt = round(length(cellModels(1).k)*cellModels(1).dt/dtDecoding);
    nht = round(length(cellModels(1).k)*cellModels(1).dt/dtDecoding);
    sumfunction = @(theBlockStructure) sum(theBlockStructure.data(:));
    spikeTimes = blockproc(spikeTimes,[dtDecoding/cellModels(1).dt,1],sumfunction);
else
    nkt = length(cellModels(1).k);
    nht = length(cellModels(1).ih);
    spikeTimes = spikeTimes;
end


kMat = zeros(nkt, numCells); %columns are the kfilter for each cell
histTermMat = zeros(slenval, numCells);
spikeVecMat = zeros(slenval, numCells);
for i = 1:numCells
    
    spikeInds = find(spikeTimes(:,i));
%     spikeVec = zeros(slenval, 1);
%     spikeVec(spikeInds) = 1;
    spikeVec = spikeTimes(1:slenval,i);
    
    %remember that kFilts end at t = 0
    initial_kt = (-nkt*cellModels(i).dt:cellModels(i).dt:-cellModels(i).dt).';
    query_kt = (-nkt*cellModels(i).dt:dtDecoding:-dtDecoding).';
    initial_ht = cellModels(i).iht;
    query_ht = (dtDecoding:dtDecoding:cellModels(i).iht(end)).';
    kMat(:,i) = interp1(initial_kt, cellModels(i).k, query_kt) * (dtDecoding/cellModels(i).dt); %this will need to get rescaled to match dtDecoding
    hTermKernelTemp(:,i) = cellModels(i).ih; % Same as basis vectors by their relative weights (cellModels(i).ihbas*cellModels(i).ihw)
    
    hTermKernel(:,i) = interp1(initial_ht,hTermKernelTemp(:,i),query_ht) * (dtDecoding/cellModels(i).dtSp);
    hTermKernel(1,i) = 0;
    
    histTermMatTemp = spikeconv_mex(spikeInds,hTermKernel,[1,slenval]);
    histTermMat(:,i) = histTermMatTemp(1:slenval) + cellModels(i).dc; % this is the unchanging hist comp + dc comp
    
    spikeVecMat(:,i) = spikeVec;
end

% currEstStim = randn(slen,1);
% stimCovMat = eye(slen);

inputParms.kMat = kMat;
inputParms.histTermMat = histTermMat;
inputParms.stimCovMatInv = inv(stimCovMat);
inputParms.stimCovMatDet = det(stimCovMat);
inputParms.spikeVecMat = spikeVecMat;
inputParms.dt = dtDecoding;

if nargin > 5
    currEstStim = initStim;
else
    currEstStim = getNoisyStim(slenval*dtDecoding,dtDecoding, 3);
end

prs0 = currEstStim;
fxn = @(prs)bayesStimDecoderLogli_Fall2019(prs,inputParms);

% opts = optimset('Gradobj','on','Hessian','on', 'maxIter', 1000, 'TolFun', 1e-8, 'TolX', 1e-8);
% opts = optimset('Gradobj','on','Hessian','on', 'display', 'iter');

opts = optimset('Gradobj','on','Hessian','on', 'MaxFunEvals', 100000, 'MaxIter', 5000, 'Display', 'off');
[prs,fval,exitflag] = fminunc(fxn,prs0,opts);

if nargout > 3 % Compute Hessian if desired
    [fval,gradval,hessian] = bayesStimDecoderLogli_Fall2019(prs, inputParms);
end

optStim = prs; 
