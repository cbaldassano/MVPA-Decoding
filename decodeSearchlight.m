function [SL_accs SL_centers SL_locs] = decodeSearchlight(bold, stimuli, num_runs, coords, xsize, ysize, zsize, radius, spacing, cost)

% This function performs searchlight decoding as a wrapper around,
% validateAndTest, iteratively grabbing a spherical ball of voxels and
% computing a decoding accuracy based on those voxels alone. This gives a
% rough localization of the voxels informative for a decoding task.

% Inputs:
% bold, stimuli, num_runs: Same as in validateAndTest.m
% coords: numvox x 3 nonnegative x,y,z integer voxel coordinates
% xsize, ysize, zsize: individual voxel dimensions in mm
% radius: searchlight radius in mm
% spacing: spacing between searchlight centers in mm
% cost: hyperparameter(s) to use for linear SVM. Typically a single number
%   is used to reduce computational time, but a list of costs (as in 
%   validateAndTest) can also be used

% Outputs:
% SL_accs: list of accuracies for each searchlight ball
% SL_centers: list of centers of each searchlight ball
% SL_locs: sets of all voxels used in each searchlight ball


% Create inverse mapping between coordinates and voxel index in bold
volDict = zeros(max(coords(:,1)+1),max(coords(:,2)+1),max(coords(:,3))+1);
for i = 1:size(coords,1)
    volDict(coords(i,1)+1,coords(i,2)+1,coords(i,3)+1) = i;
end

% Generate searchlight center coords
xmax = max(coords(:,1)); ymax = max(coords(:,2)); zmax = max(coords(:,3));
xmin = min(coords(:,1)); ymin = min(coords(:,2)); zmin = min(coords(:,3));
[x y z] = meshgrid(round(xmin:spacing/xsize:xmax),round(ymin:spacing/ysize:ymax),round(zmin:spacing/zsize:zmax));
x = x(:); y = y(:); z = z(:);

% Represent searchlight ball in terms of linear offsets
ball = [];
ballPos = zeros(0,3);
for xi = -1*ceil(radius/xsize):ceil(radius/xsize)
    for yi = -1*ceil(radius/ysize):ceil(radius/ysize)
        for zi = -1*ceil(radius/zsize):ceil(radius/zsize)
            if ((xi*xsize)*(xi*xsize)+(yi*ysize)*(yi*ysize)+(zi*zsize)*(zi*zsize) <= radius*radius)
                ball = [ball (1 + (xi+1-1) + (yi+1-1)*size(volDict,1) + (zi+1-1)*size(volDict,1)*size(volDict,2))];
                ballPos(end+1,:) = [xi yi zi];
            end
        end
    end
end

SL_accs = [];
SL_centers = zeros(0,3);
SL_locs = cell(0,1);
for i = 1:length(x)
    xcenter = x(i); ycenter = y(i); zcenter = z(i);

    % Compute valid voxels within current search ball
    currBall = ball+xcenter+ycenter*size(volDict,1) + zcenter*size(volDict,1)*size(volDict,2);
    currPos = [ballPos(:,1)+xcenter ballPos(:,2)+ycenter ballPos(:,3)+zcenter];
    currBall = currBall(~(currPos(:,1) < xmin | currPos(:,2) < ymin | currPos(:,3) < zmin ...
        | currPos(:,1) > xmax | currPos(:,2) > ymax | currPos(:,3) > zmax));
    searchInds = volDict(currBall);
    searchInds = searchInds(searchInds ~= 0);


    if (~isempty(searchInds))
        SL_accs = [SL_accs validateAndTest(bold(searchInds,:), stimuli, num_runs, [], [], [], [], [], [], [], [], [], cost)];
        SL_locs{end+1,1} = coords(searchInds,1:3);
        SL_centers = [SL_centers; [xcenter ycenter zcenter]];
    end

end

end