function [accs confmats] = validateAndTest(bold, stimuli, NUM_RUNS, TRS_PER_RUN, trainingStimuli, trainingCategories, testingStimuli, testingCategories, testingConditions, valFlags, excludeRun, blockVoting, numvox, quiet)

% Note: this function requires libsvm http://www.csie.ntu.edu.tw/~cjlin/libsvm/

% Inputs:
% bold: numvox x timepoints 
% stimuli: timepoints x 1 label vector, with distinct categories labeled 1,2,...
%   These should have already been shifted to account for hemodynamic lag
%   No-stimuli timepoints should be labeled 0 (for use in voxel selection)
% NUM_RUNS, TRS_PER_RUN: number of independent runs and length of each
%     If you want to leave out multiple runs together (e.g. if it takes multiple runs
%     to display all categories) then this should be the number and length of the pseudoruns
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
% quiet: set to 1 to suppress extra output

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