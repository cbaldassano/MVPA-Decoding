function [accs confmats] = validateAndTest(bold, stimuli, num_runs, trainingStimuli, trainingCategories, testingStimuli, testingCategories, testingConditions, valFlags, excludeRun, blockVoting, numvox, costs)

% Note: this function requires libsvm http://www.csie.ntu.edu.tw/~cjlin/libsvm/

% Inputs:

% Required:
% bold: numvox x timepoints 
% stimuli: timepoints x 1 label vector, with distinct categories labeled
%   with distinct values (do not have to be sequential)
%   These should have already been shifted to account for hemodynamic lag
%   No-stimuli timepoints should be labeled 0 (for use in voxel selection)
% num_runs: number of independent runs (assumed to be concatenated together)
%     If you want to leave out multiple runs together (e.g. if it takes multiple runs
%     to display all categories) then this should be the number of the pseudoruns

% Advanced (Optional):
% trainingStimuli: the set of labels for training (same key as stimuli vector)
% trainingCategories: the classes of the training stimuli, labeled 1,2,... (NOT the same as stimuli vector)
% testing Stimuli: the set of labels for testing (same key as stimuli vector)
% testingCategories: the classes of the testing stimuli, labeled 1,2,... (NOT the same as stimuli vector)
% testingConditions: used to separate testing accuracies into different groups labeled 1,2,...
% valFlags: which testing conditions to use for tuning hyperparameters (0/1)
% excludeRun: whether to disallow testing and training in the same run
% blockVoting: whether each block should take a majority vote to determine category
% numvox: number of voxels to use (chosen by non-zero vs. zero timepoint z-score),
%     Use -1 to use all voxels
%     Can also be given as a fraction [0 1]
% costs: list of cost hyperparameters to try during cross-validation


% Outputs:
% accs: a vector of accuracies for each testing condition
% confmats (optional): a num conditions x num test cats x num test cats
% matrix showing counts of actual categories (dim 2) vs predicted categories (dim 3)
% These counts are in TRs, or in blocks if blockVoting is enabled
% If there is only one condition, the first dimension is removed


% Examples
% Assume we have 4 stimuli:
%   1: Men with hats
%   2: Men without hats
%   3: Women with hats
%   4: Women without hats
% 
% Possible choices for (trainingStimuli, trainingCategories,
% testingStimuli, testingCategories, testingConditions, valFlags)
%
% Hat classification:
%   ([1 2 3 4],[1 2 1 2],[1 2 3 4],[1 2 1 2],[1 1 1 1],[1 1 1 1])
% Hat classification on women only:
%   ([3 4],[1 2],[3 4],[1 2],[1 1],[1 1])
% Gender classification:
%   ([1 2 3 4],[1 1 2 2],[1 2 3 4],[1 1 2 2],[1 1 1 1],[1 1 1 1])
% Male hat classifier with cross-decoding to female hat classification
%   ([1 2],[1 2],[1 2 3 4],[1 2 1 2],[1 1 2 2],[1 1 0 0])

% In old versions of Matlab (pre-2009b) you will need to replace the tilde
% dummy variables (~) with a named dummy variable (any name not in use)


% Set defaults for inputs
if (~exist('trainingStimuli','var') || ~exist('trainingCategories','var') || ...
     isempty(trainingStimuli) || isempty(trainingCategories))
    trainingStimuli = unique(stimuli(stimuli>0));
    trainingCategories = 1:length(trainingStimuli);
end
if (~exist('testingStimuli','var') || ~exist('testingCategories','var') || ...
     isempty(testingStimuli) || isempty(testingCategories))
    testingStimuli = trainingStimuli;
    testingCategories = trainingCategories;
end
if (~exist('testingConditions','var') || isempty(testingConditions))
    testingConditions = ones(1,length(testingStimuli));
end
if (~exist('valFlags','var') || isempty(valFlags))
    valFlags = ones(1,length(testingConditions));
end
if (~exist('excludeRun','var') || isempty(excludeRun))
    excludeRun = 1;
end
if (~exist('blockVoting','var') || isempty(blockVoting))
    blockVoting = 0;
end
if (~exist('numvox','var') || isempty(numvox))
    numvox = -1;
end
if (~exist('costs','var') || isempty(costs))
    costs = logspace(-5,1.5,20);
end

% If training and testing stimuli overlap, training stimuli cannot be drawn
% from the testing run
assert(excludeRun || isempty(intersect(trainingStimuli,testingStimuli)));

runNum = cumsum(repmat([1;zeros(length(stimuli)/num_runs-1,1)],num_runs,1));

if (numvox <= 0)
    numvox = size(bold,1);
elseif (numvox <= 1)
    numvox = round(numvox*size(bold,1));
end
if (numvox > size(bold,1))
    numvox = size(bold,1);
end


confmats = zeros(length(unique(testingConditions)),max(testingCategories),max(testingCategories));
testingRunAcc = NaN(num_runs,length(unique(testingConditions)));
for testingRun = 1:num_runs
    % First, choose best hyperparameter by holding out validationRun
    if (length(costs)==1)
        meancost = costs(1);
    else
        bestCosts = -1*ones(num_runs,1);
        for validationRun = 1:num_runs
            if (testingRun == validationRun)
                continue;
            end

            % Construct training set
            trainingInds = cell(max(trainingCategories),1);
            for c = 1:length(trainingInds)
                if (excludeRun)
                    trainingInds{c} = find(ismember(stimuli,trainingStimuli(trainingCategories==c)) & ...
                                          ~ismember(runNum,[testingRun validationRun]));
                else
                    trainingInds{c} = find(ismember(stimuli,trainingStimuli(trainingCategories==c)) & ...
                                          ~ismember(runNum,testingRun));
                end
            end

            % Construct testing set
            testingInds = cell(max(testingCategories),1);
            for c = 1:length(testingInds)
               testingInds{c} = find(ismember(stimuli,testingStimuli(testingCategories==c & valFlags)) & ...
                                     runNum == validationRun);
            end

            assert(isempty(intersect(cell2mat(trainingInds),cell2mat(testingInds))));

            % Perform voxel selection
            if (numvox < size(bold,1))
                voxelZ = zeros(size(bold,1),1);
                for voxel = 1:size(bold,1)
                   restingMean = mean(bold(voxel,stimuli == 0 & ~ismember(runNum,[testingRun validationRun])));
                   activeMean = mean(bold(voxel,cell2mat(trainingInds)));
                   activeStd = std(bold(voxel,cell2mat(trainingInds)));
                   voxelZ(voxel) = abs(activeMean - restingMean)/activeStd;
                end

                [~, indNoise] = sort(voxelZ,'descend');
                processedBold = bold(indNoise(1:numvox),:);
            else
                processedBold = bold;
            end

            % Try all possible cost hyperparameters
            accCV = zeros(length(costs),1);
            c = 1;
            for cost = costs

                % Train and predict
                [model means scaling] = svmtrain_wrapper(processedBold,cost,trainingInds);

                % Note that block voting is always disabled (the last
                % argument to svmpredict_wrapper) during validation,
                % because it introduces more variability in the
                % results)
                accCV(c) = svmpredict_wrapper(processedBold,model,means,scaling,testingInds,0);

                c = c+1;
            end
            clear cost

            % Smooth the accuracy matrix before picking a best choice of parameters
            % This is experimental, but seems to work well in choosing robust
            % parameters
            accCV(:) = smooth(accCV(:),5,'sgolay');
            [~,bestCostInd] = max(accCV(:));
            bestCosts(validationRun) = costs(bestCostInd);
        end
        
        meancost = exp(mean(log(bestCosts(bestCosts>0))));
    end
    
    % Perform voxel selection, using all training runs
    if (numvox < size(bold,1))
        voxelZ = zeros(size(bold,1),1);
        for voxel = 1:size(bold,1)
           restingMean = mean(bold(voxel,stimuli == 0 & ~ismember(runNum,testingRun)));
           activeMean = mean(bold(voxel,cell2mat(trainingInds)));
           activeStd = std(bold(voxel,cell2mat(trainingInds)));
           voxelZ(voxel) = abs(activeMean - restingMean)/activeStd;
        end

        [~, indNoise] = sort(voxelZ,'descend');
        processedBold = bold(indNoise(1:numvox),:);
    else
        processedBold = bold;
    end
       
    % Construct training set
    trainingInds = cell(max(trainingCategories),1);
    for c = 1:length(trainingInds)
        if (excludeRun)
            trainingInds{c} = find(ismember(stimuli,trainingStimuli(trainingCategories==c)) & ...
                                  ~ismember(runNum,testingRun));
        else
            trainingInds{c} = find(ismember(stimuli,trainingStimuli(trainingCategories==c)));
        end
    end

    %Train classifier
    [model means scaling] = svmtrain_wrapper(processedBold,meancost,trainingInds);

    % Test independently on each of the testing conditions
    for condition = unique(testingConditions)
        testingInds = cell(max(testingCategories),1);
        for c = 1:length(testingInds)
           testingInds{c} = find(ismember(stimuli,testingStimuli(testingCategories==c & testingConditions == condition)) & ...
                                 runNum == testingRun);
        end

        if (isempty(cell2mat(testingInds)))
           continue;
        end

        assert(isempty(intersect(cell2mat(trainingInds),cell2mat(testingInds))))

        [accuracy condConfmat] = svmpredict_wrapper(processedBold,model,means,scaling,testingInds,blockVoting);
        confmats(condition,:,:) = confmats(condition,:,:) + reshape(condConfmat,1,size(confmats,2),size(confmats,3));

        testingRunAcc(testingRun,condition) = accuracy;
    end
end

accs = zeros(1,length(unique(testingConditions)));
for condition = 1:size(accs,2)
    accs(condition) = mean(testingRunAcc(~isnan(testingRunAcc(:,condition)),condition));
end

if (size(confmats,1) == 1)
    confmats = squeeze(confmats);
end
end
