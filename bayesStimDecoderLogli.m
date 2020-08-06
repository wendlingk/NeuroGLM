% function [neglogli, grad, hessian] = bayesStimDecoderLogli(prs, inputParms)
function [neglogli, grad, hessian] = bayesStimDecoderLogli(prs, inputParms)


kMat = inputParms.kMat;
histTermMat = inputParms.histTermMat;
stimCovMatInv = inputParms.stimCovMatInv;
stimCovMatDet = inputParms.stimCovMatDet;
spikeVecMat = inputParms.spikeVecMat;
dt = inputParms.dt; %of the decoding (in msec)
%dt = dt*.001; %converts dt into seconds

numCells = size(kMat,2);
slen = size(histTermMat,1);

currStim = prs;

neglogli = 0;
grad = zeros(slen, 1);
hessian = zeros(slen, slen);

tPts = (inputParms.dt:inputParms.dt:slen*inputParms.dt).';

for i = 1:numCells
    
    currSpikeVec = spikeVecMat(:,i);
    kVec = kMat(:,i);
    kVec(length(kVec)+1:slen+ length(kVec)) = zeros(slen,1);
    cellKMat = triu(toeplitz(kVec));
    cellKMat = cellKMat(1:slen, size(kMat,1):slen+size(kMat,1)-1);
    
    stimCompInput = cellKMat*currStim;
    fullFxnInput = stimCompInput + histTermMat(:,i);
    
    [condInt, dcondInt, ddcondInt, logCondInt, dlogCondInt, ddlogCondInt] = expfunAndLog(fullFxnInput);
    
    cellNeglogli = -currSpikeVec'*logCondInt + sum(condInt)*dt;
    cellGrad = cellKMat'*(dcondInt*dt - currSpikeVec.*dlogCondInt);
    cellHessian = cellKMat*diag(ddcondInt*dt - currSpikeVec.*ddlogCondInt)*cellKMat';
%     
    neglogli = neglogli + cellNeglogli;
    grad = grad + cellGrad;
    hessian = hessian + cellHessian;
end

if stimCovMatDet ~= -1
    priorNeglogli = .5*(currStim'*stimCovMatInv*currStim);% + (slen/2)*log(2*pi) + .5*log(stimCovMatDet);
    priorGrad = currStim'*stimCovMatInv;
    priorHessian = stimCovMatInv;    
else
    priorNeglogli = 0;%(.5*slen*log(2*pi) + .5*log(priorCovMatDet)+ .5*currEstStim'*priorInvCovMat*currEstStim); %already negated
    priorGrad = 0;%currEstStim'*priorInvCovMat;
    priorHessian = 0;%priorInvCovMat';
end


neglogli = neglogli + priorNeglogli;
grad = grad + priorGrad.';
hessian = hessian + priorHessian;


%     stimCompCheck = sameconv(currStim,kMat(:,i));
    
