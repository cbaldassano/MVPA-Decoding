function example()

% Create a four-stimulus experiment
% (we assume hemodynamic lag has already been accounted for)
NUM_RUNS = 3;
TRS_PER_RUN = 126;
stim_onerun = zeros(TRS_PER_RUN,1);
stim_onerun(15:28) = 1;
stim_onerun(43:56) = 2;
stim_onerun(71:84) = 3;
stim_onerun(99:112) = 4;
stimuli = repmat(stim_onerun,NUM_RUNS,1);

% ROI Decoding

% Generate Gaussian data for each class
Nvox = 10;
Nnoisevox = 2;
noiseStd = 1.5;
stimMeans = rand(Nvox-Nnoisevox, 4);

bold = zeros(Nvox, length(stimuli));
bold(1:(Nvox-Nnoisevox), stimuli==0) = noiseStd * rand(Nvox-Nnoisevox, sum(stimuli==0));
for i = 1:4
    bold(1:(Nvox-Nnoisevox), stimuli==i) = ...
        bsxfun(@plus, noiseStd * rand(Nvox-Nnoisevox, sum(stimuli==i)), ...
                     stimMeans(:,i));
end
bold((Nvox-Nnoisevox+1):Nvox,:) = noiseStd * rand(Nnoisevox,length(stimuli));

disp('ROI Decoding Results:');
disp(' ');

% Simple four-way classification with confusion matrix (no voxel selection)
[accs confmat] = validateAndTest(bold, stimuli, NUM_RUNS);
disp(['Four-way classification: ' num2str(accs) '%']);
disp('Confusion matrix:');
disp([9 'Pred1' 9 'Pred2' 9 'Pred3' 9 'Pred4']);
disp(['True1' 9 num2str(confmat(1,1)) 9 num2str(confmat(1,2)) 9 num2str(confmat(1,3)) 9 num2str(confmat(1,4))]);
disp(['True2' 9 num2str(confmat(2,1)) 9 num2str(confmat(2,2)) 9 num2str(confmat(2,3)) 9 num2str(confmat(2,4))]);
disp(['True3' 9 num2str(confmat(3,1)) 9 num2str(confmat(3,2)) 9 num2str(confmat(3,3)) 9 num2str(confmat(3,4))]);
disp(['True4' 9 num2str(confmat(4,1)) 9 num2str(confmat(4,2)) 9 num2str(confmat(4,3)) 9 num2str(confmat(4,4))]);
disp(' ');

% Classify first two stimuli vs. last two stimuli
accs = validateAndTest(bold, stimuli, NUM_RUNS, [1 2 3 4], [1 1 2 2]);
disp(['First two vs last two: ' num2str(accs) '%']);
disp(' ');

% Classify first vs. second stimuli, then apply to third vs. fourth (with voxel selection)
accs = validateAndTest(bold, stimuli, NUM_RUNS, [1 2], [1 2], [1 2 3 4], [1 2 1 2], [1 1 2 2], [1 1 0 0], [], [], Nvox-Nnoisevox);
disp(['First vs second: ' num2str(accs(1)) '%,']);
disp(['   Applying learned classifier to third vs. fourth: ' num2str(accs(2)) '%']);
disp(' ');
disp(' ');




% Searchlight Decoding

%  Generate Gaussian data for each class
Nvox = 100;
noiseStd = 1.5;
stimMeans = rand(Nvox, 4);

bold = zeros(Nvox, length(stimuli));
bold(:, stimuli==0) = noiseStd * rand(Nvox, sum(stimuli==0));
for i = 1:4
    bold(:, stimuli==i) = ...
        bsxfun(@plus, noiseStd * rand(Nvox, sum(stimuli==i)), ...
                     stimMeans(:,i));
end

coords = [mod((1:100)-1,10)' mod(floor(((1:100)-1)/10),10)' zeros(100,1)];
xsize = 1; ysize = 1; zsize = 1;
radius = 2.5;
spacing = 4;
cost = 1;

disp('Searchlight Decoding Results:');
disp(' ');
% Simple four-way classification at each location
[SL_accs SL_centers SL_locs] = decodeSearchlight(bold, stimuli, NUM_RUNS, coords, xsize, ysize, zsize, radius, spacing, cost);
for i = 1:length(SL_accs)
    disp(['(' num2str(SL_centers(i,1)) ',' num2str(SL_centers(i,2)) '): ' num2str(SL_accs(i)) '%']);
end
end