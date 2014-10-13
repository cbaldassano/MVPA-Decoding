function [predictAcc confmat] = svmpredict_wrapper(bold,model,means,scaling,testingInds,blockVoting)

numCats = length(testingInds);
testingData = cell(numCats,1);
labels = [];
for c = 1:numCats
    testingData{c} = bold(:,testingInds{c})';
    labels = [labels;c*ones(length(testingInds{c}),1)];
    
    for j = 1:size(testingData{c},2)
        testingData{c}(:,j) = (testingData{c}(:,j) - means(j))/scaling(j);
    end
end
allTestingData = cell2mat(testingData);

[predictedlabel, accuracy] = svmpredict(labels, allTestingData, model);
if (~blockVoting)
    predictAcc = accuracy(1);
    if (nargout > 1)
        confmat = zeros(numCats,numCats);
        for c1 = 1:numCats
            for c2 = 1:numCats
                confmat(c1,c2) = sum(labels==c1 & predictedlabel == c2);
            end
        end
    end
else
    labelInd = 1;
    predictionsByClass = cell(numCats,1);
    for c = 1:numCats
        predictionsByClass{c} = predictedlabel(labelInd:labelInd+length(testingInds{c})-1);
        labelInd = labelInd + length(testingInds{c});
    end
    
    blockAcc = [];
    blockPred = [];
    blockTrue = [];
    for c = 1:numCats
        if (isempty(testingInds{c}))
            continue;
        end
        blockStart = 1;
        i = 2;
        while 1
            while ((i <= length(testingInds{c})) && (testingInds{c}(i-1)+1 == testingInds{c}(i)))
                i = i + 1;  
            end
            
            blockPred = [blockPred mode(predictionsByClass{c}(blockStart:i-1))];
            blockTrue = [blockTrue c];
            blockAcc = [blockAcc 100*(blockPred(end)==c)];
            
            if (i > length(testingInds{c}))
                break;
            else
                blockStart = i;
                i = blockStart + 1;
            end
        end
    end
    
    predictAcc = mean(blockAcc);
    
    if (nargout > 1)
        confmat = zeros(numCats,numCats);
        for c1 = 1:numCats
            for c2 = 1:numCats
                confmat(c1,c2) = sum(blockTrue==c1 & blockPred == c2);
            end
        end
    end
    
end
    