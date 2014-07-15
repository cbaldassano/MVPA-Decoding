function [model means scaling] = svmtrain_wrapper(bold,cost,trainingInds)

trainingData = cell(length(trainingInds),1);
labels = [];
for c = 1:length(trainingInds)
    trainingData{c} = bold(:,trainingInds{c})';
    labels = [labels;c*ones(length(trainingInds{c}),1)];
end
allTrainingData = cell2mat(trainingData);

scaling = zeros(size(allTrainingData,2),1);
means = zeros(size(allTrainingData,2),1);
for i=1:size(allTrainingData,2)
    scaling(i) = std(allTrainingData(:,i));
    means(i) = mean(allTrainingData(:,i));
    
    for c = 1:length(trainingInds)
        trainingData{c}(:,i) = (trainingData{c}(:,i) - means(i))/scaling(i);
    end
end
allTrainingData = cell2mat(trainingData);

model = svmtrain(labels, allTrainingData, ['-c ' num2str(cost) ' -t 0']);

end