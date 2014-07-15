function predictAcc = svmpredict_wrapper(bold,model,means,scaling,testingInds,blockVoting)

testingData = cell(length(testingInds),1);
labels = [];
for c = 1:length(testingInds)
    testingData{c} = bold(:,testingInds{c})';
    labels = [labels;c*ones(length(testingInds{c}),1)];
    
    for j = 1:size(testingData{c},2)
        testingData{c}(:,j) = (testingData{c}(:,j) - means(j))/scaling(j);
    end
end
allTestingData = cell2mat(testingData);

[predictedlabel, accuracy, ~] = svmpredict(labels, allTestingData, model);
if (~blockVoting)
    predictAcc = accuracy(1);
else
    labelInd = 1;
    predictionsByClass = cell(length(testingInds),1);
    for c = 1:length(testingInds)
        predictionsByClass{c} = predictedlabel(labelInd:labelInd+length(testingInds{c})-1);
        labelInd = labelInd + length(testingInds{c});
    end
    
    blockAcc = [];
    for c = 1:length(testingInds)
        if (isempty(testingInds{c}))
            continue;
        end
        blockStart = 1;
        i = 2;
        while 1
            while ((i <= length(testingInds{c})) && (testingInds{c}(i-1)+1 == testingInds{c}(i)))
                i = i + 1;  
            end
            
            blockAcc = [blockAcc 100*(mode(predictionsByClass{c}(blockStart:i-1))==c)];
            
            if (i > length(testingInds{c}))
                break;
            else
                blockStart = i;
                i = blockStart + 1;
            end
        end
    end
    
    predictAcc = mean(blockAcc);
    
end
    