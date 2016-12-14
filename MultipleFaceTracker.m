classdef MultipleFaceTracker < handle
    properties        
        showUnknownClass = 1;        
        faceClassifier;                        
        PointTracker; % Track points                
        FBboxes = []; % Store bounding face box in the follwoing form: [x y width heigth]                
        BoxIds = []; % BoxIds M-by-1 array containing ids associated with each bounding box                
        Points = []; % Points M-by-2 matrix containing tracked points from all objects                
        PointIds = []; %This keeps track of which point belongs to which face.                
        NextId = 1; % The next new object will have this id.                
        BoxScores = []; % BoxScores M-by-1 array. Low box score means that we probably lost the object.        
        personLabel = {}; %Store classified label for each face so that it's not required to be computed again for each call                
    end
    
    methods
        % Constructor
        function this = MultipleFaceTracker()
        
            this.PointTracker = vision.PointTracker('MaxBidirectionalError', 2);
        end
        
        %for the deteced face, add the bounding box and verifies whether
        %it's new detected face box or an existing one.
        function classifyFaces(this, I, boxFrame, framergb)        
            
            [fx,fy,fz] = size(framergb);
            
            changed = 0;
            %Iterate through all the deteced faces and if's a new face, add
            %bounding box and add to tracking else just track them.
            for i = 1:size(boxFrame, 1)
                % Determine if the detection belongs to one of the existing
                % face.
                boxIdx = this.findMatchingBox(boxFrame(i, :));
                
                if isempty(boxIdx)                    
                    
                    x = boxFrame(i,2);
                    h = boxFrame(i,3);
                    y = boxFrame(i,1);
                    w = boxFrame(i,4);                   
                    
                    faceFrame = framergb(x:x+h, y:y+w, :);
                    scaleFac = 150/size(faceFrame, 1);
                    faceFrame = imresize(faceFrame, scaleFac);                                       
                    
                    queryFeatures = extractHOGFeatures(faceFrame);
                    [predictedLabel,NegLoss,PBScore,Posterior] = predict(this.faceClassifier, queryFeatures);
                    if max(Posterior) < 0.9                        
                        if ~this.showUnknownClass 
                            continue;
                        end
                        this.personLabel{length(this.personLabel)+1} = 'unknown';
                    else
                        this.personLabel(length(this.personLabel)+1) = predictedLabel(1);
                    end
                    
                    %figure, imshow(faceFrame), title([char(predictedLabel), num2str(prob), num2str(cost)]);
                    %It's a new bounding box
                    this.FBboxes = [this.FBboxes; boxFrame(i, :)];
                    
                    points = detectMinEigenFeatures(I, 'ROI', boxFrame(i, :));
                    points = points.Location;
                    this.BoxIds(end+1) = this.NextId;
                    idx = ones(size(points, 1), 1) * this.NextId;
                    this.PointIds = [this.PointIds; idx];
                    this.NextId = this.NextId + 1;
                    this.Points = [this.Points; points];
                    this.BoxScores(end+1) = 1;
                    
                    changed = changed + 1;
                    
                else    %An existing face.
                    
                    %find the corresponding label for the face
                    label = this.personLabel(this.BoxIds == boxIdx);
                    % Delete the matched bounding box
                    currentBoxScore = this.deleteBox(boxIdx);                                        
                                        
                    x = boxFrame(i,2);
                    h = boxFrame(i,3);
                    y = boxFrame(i,1);
                    w = boxFrame(i,4);                       

                    faceFrame = framergb(x:x+h, y:y+w, :);
                    scaleFac = 150/size(faceFrame, 1);
                    faceFrame = imresize(faceFrame, scaleFac);                        

                    queryFeatures = extractHOGFeatures(faceFrame);
                    [predictedLabel,NegLoss,PBScore,Posterior] = predict(this.faceClassifier, queryFeatures);

                    if max(Posterior) < 0.9
                        label = {'unknown'};                            
                        if ~this.showUnknownClass 
                            continue;
                        end
                    else
                        label = predictedLabel(1);
                    end
                        
                        
                    
                    % Replace with new box
                    this.FBboxes = [this.FBboxes; boxFrame(i, :)];
                    
                    % Re-detect the points. This is how we replace the
                    % points, which invariably get lost as we track.
                    points = detectMinEigenFeatures(I, 'ROI', boxFrame(i, :));
                    points = points.Location;
                    this.BoxIds(end+1) = boxIdx;
                    idx = ones(size(points, 1), 1) * boxIdx;
                    this.PointIds = [this.PointIds; idx];
                    this.Points = [this.Points; points];                    
                    this.BoxScores(end+1) = currentBoxScore + 1;
                    
                    this.personLabel(length(this.personLabel)+1) = label;    
                    changed = changed + 1;
                end
            end
            
            % Determine which faces are no longer tracked.
            minBoxScore = 0;
            this.BoxScores(this.BoxScores < 3) = ...
                this.BoxScores(this.BoxScores < 3) - 0.5;
            boxesToRemoveIds = this.BoxIds(this.BoxScores < minBoxScore);
            while ~isempty(boxesToRemoveIds)
                this.deleteBox(boxesToRemoveIds(1));
                boxesToRemoveIds = this.BoxIds(this.BoxScores < minBoxScore);
            end
            
            % Update the point tracker.
            if~isempty(this.Points) || changed > 0
                if this.PointTracker.isLocked()
                    this.PointTracker.setPoints(this.Points);
                else
                    this.PointTracker.initialize(this.Points, I);
                end
            end                        
            
        end
                        
        function track(this, I)
        % Track the faces by updating the points and the object bounding boxes.
            if isempty(this.Points)
                return;
            end
            [newPoints, isFound] = this.PointTracker.step(I);
            this.Points = newPoints(isFound, :);
            this.PointIds = this.PointIds(isFound);
            generateNewBoxes(this);
            if ~isempty(this.Points)
                this.PointTracker.setPoints(this.Points);
            end            
        end    
                
        function boxIdx = findMatchingBox(this, box)
        % Determine which tracked object (if any) the new detection belongs to. 
            boxIdx = [];
            for i = 1:size(this.FBboxes, 1)
                area = rectint(this.FBboxes(i,:), box);                
                if area > 0.2 * this.FBboxes(i, 3) * this.FBboxes(i, 4) 
                    boxIdx = this.BoxIds(i);
                    return;
                end
            end           
        end
                
        function currentScore = deleteBox(this, boxIdx)            
        % Delete object.
            this.FBboxes(this.BoxIds == boxIdx, :) = [];
            this.Points(this.PointIds == boxIdx, :) = [];
            this.PointIds(this.PointIds == boxIdx) = [];
            currentScore = this.BoxScores(this.BoxIds == boxIdx);
            this.BoxScores(this.BoxIds == boxIdx) = [];
            this.personLabel(this.BoxIds == boxIdx) = [];
            this.BoxIds(this.BoxIds == boxIdx) = [];            
        end
                
        function generateNewBoxes(this)  
        % Get bounding boxes for each object from tracked points.
            oldBoxIds = this.BoxIds;
            oldScores = this.BoxScores;
            oldPersonLabel = this.personLabel;
            this.BoxIds = unique(this.PointIds);
            numBoxes = numel(this.BoxIds);
            this.FBboxes = zeros(numBoxes, 4);
            this.BoxScores = zeros(numBoxes, 1);
            this.personLabel = {};
            for i = 1:numBoxes
                points = this.Points(this.PointIds == this.BoxIds(i), :);
                
                %Compute face bounding box
                x1 = min(points(:, 1));
                y1 = min(points(:, 2));
                x2 = max(points(:, 1));
                y2 = max(points(:, 2));
                newBox = [x1 y1 x2 - x1 y2 - y1];
                
                this.FBboxes(i, :) = newBox;
                this.BoxScores(i) = oldScores(oldBoxIds == this.BoxIds(i));
                this.personLabel(i) = oldPersonLabel(oldBoxIds == this.BoxIds(i));
            end
        end 
    end
end