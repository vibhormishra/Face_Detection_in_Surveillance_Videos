% Load Image Information from the database
faceDb = imageSet('cv_imageset', 'recursive');

strSize = [0 0];

% Extract HOG features for training set
features = zeros(size(faceDb,2)* faceDb(1).Count,10404);
featureCount = 1;
for i=1:size(faceDb,2)
    for j=1:faceDb(i).Count
        features(featureCount,:) = extractHOGFeatures(read(faceDb(i),j));
        label{featureCount} = faceDb(i).Description;
        featureCount = featureCount + 1;
        
        % Update progress every 200th image.
        if (mod(j, 200) == 0|| j ==  faceDb(i).Count)
            if j > 200
                fprintf(repmat('\b',1,strSize(1,2)));
            end
            str = sprintf('Image set for class: [%s] done. %d / %d (%.0f%%)', faceDb(i).Description, j, faceDb(i).Count, (j / faceDb(i).Count) * 100);
            fprintf('%s\r',str)
            strSize = size(str)+1;        
        end
    
    end
    % Update progress every image class.
    str = sprintf('%%%%%%%%%Class done: [%s] Remaining: %d / %d (%.0f%%)%%%%%%%%%\n', faceDb(i).Description, i, size(faceDb,2), (i / size(faceDb,2)) * 100);
    fprintf('%s\r',str)
        
end


% create class classifier using fitcecoc
faceClassifier = fitcecoc(features, label, 'FitPosterior', 1);

%Save the trained classifier
save('faceClassifier.mat','faceClassifier');