function results = validateAndTest(bold, stimuli, NUM_RUNS, TRS_PER_RUN, trainingStimuli, trainingCategories, testingStimuli, testingCategories, testingConditions, valFlags, excludeRun,blockVoting,numvox,quiet)

% Note: this function requires libsvm http://www.csie.ntu.edu.tw/~cjlin/libsvm/

% Inputs:
% bold: numvox x timepoints 
% stimuli: timepoints x 1 label vector, with distinct categories labeled 1,2,...
%   These should have already been shifted to account for hemodynamic lag
% NUM_RUNS, TRS_PER_RUN: number of independent runs and length of each
%     If you want to leave out multiple runs together (e.g. if it takes multiple runs
%     to display all categories) then this should be the number of length of the pseudoruns
% trainingStimuli: the set of labels for training (same key as stimuli vector)
% trainingCategories: the classes of the training stimuli, labeled 1,2,... (NOT the same as stimuli vector)
% testing Stimuli: the set of labels for testing (same key as stimuli vector)
% testingCategories: the classes of the testing stimuli, labeled 1,2,... (NOT the same as stimuli vector)
% testingConditions: used to separate testing accuracies into different groups labeled 1,2,...
% valFlags: which testing conditions to use for tuning hyperparameters (0/1)
% excludeRun: whether to disallow testing and training in the same run
% blockVoting: whether each block should take a majority vote to determine category
% numvox: number of voxels to use (chosen by visual response z-score), use -1 to use all voxels
%     Can also be given as a fraction [0 1]
% quiet: set to 1 to suppress extra output

% Output: a vector of accuracies for each testing condition

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

% Copyright (c) 2014, Christopher Baldassano, Stanford University
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in the
%       documentation and/or other materials provided with the distribution.
%     * Neither the name of Stanford University nor the
%       names of its contributors may be used to endorse or promote products
%       derived from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL CHRISTOPHER BALDASSANO BE LIABLE FOR ANY
% DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% If training and testing stimuli overlap, training stimuli cannot be drawn
% from the testing run
assert(excludeRun || isempty(intersect(trainingStimuli,testingStimuli)));

runNum = cumsum(repmat([1;zeros(TRS_PER_RUN-1,1)],NUM_RUNS,1));

%Costs to try in cross-validation
costs = logspace(-5,1.5,20);

if (numvox <= 0)
    numvox = size(bold,1);
elseif (numvox <= 1)
    numvox = round(numvox*size(bold,1));
end
if (numvox > size(bold,1))
    numvox = size(bold,1);
end

testingRunAcc = zeros(NUM_RUNS,length(unique(testingConditions)));
cvParams = zeros(NUM_RUNS,NUM_RUNS);
for testingRun = 1:NUM_RUNS
    if (~quiet)
        disp([num2str(testingRun) '/' num2str(NUM_RUNS) ' starting']);
    end
    validRunsForCondition = zeros(length(unique(testingConditions)),1);
   for validationRun = 1:NUM_RUNS
        if (testingRun == validationRun)
            continue;
        end
        
        % First, choose best parameters by holding out validationRun
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
                               
       testingInds = cell(max(testingCategories),1);
       for c = 1:length(testingInds)
           testingInds{c} = find(ismember(stimuli,testingStimuli(testingCategories==c & valFlags)) & ...
                                 runNum == validationRun);
       end

       assert(isempty(intersect(cell2mat(trainingInds),cell2mat(testingInds))));

        % Try all combination of cross-validation variables
        accCV = zeros(length(costs),1);
        
        voxelZ = zeros(size(bold,1),1);
        for voxel = 1:size(bold,1)
           restingMean = mean(bold(voxel,stimuli == 0 & ~ismember(runNum,[testingRun validationRun])));
           activeMean = mean(bold(voxel,cell2mat(trainingInds)));
           activeStd = std(bold(voxel,cell2mat(trainingInds)));
           voxelZ(voxel) = abs(activeMean - restingMean)/activeStd;
        end

        [~, indNoise] = sort(voxelZ,'descend');

        processedBold = bold(indNoise(1:numvox),:);

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
       [~,bestCost] = max(accCV(:));

       cvParams(testingRun,validationRun) = costs(bestCost);
       
       if (~quiet && (bestCost == 1 || bestCost == length(costs)))
           disp(['Selected cost ' num2str(costs(bestCost)) ', consider changing range']);
       end
       
       %Now, test on testing run
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
        [model means scaling] = svmtrain_wrapper(processedBold,costs(bestCost),trainingInds);
    
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

            accuracy = svmpredict_wrapper(processedBold,model,means,scaling,testingInds,blockVoting);

            testingRunAcc(testingRun,condition) = testingRunAcc(testingRun,condition) + accuracy;
            validRunsForCondition(condition) = validRunsForCondition(condition)+1;
        end
   end
   for condition = unique(testingConditions)
        testingRunAcc(testingRun,condition) = testingRunAcc(testingRun,condition)/validRunsForCondition(condition);   end
end

results = zeros(1,length(unique(testingConditions)));
for condition = 1:size(results,2)
    results(condition) = mean(testingRunAcc(~isnan(testingRunAcc(:,condition)),condition));
end
