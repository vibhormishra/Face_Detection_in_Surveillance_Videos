clear;

%Initialize face tracker
faceTracker = MultipleFaceTracker;

if ~exist('faceClassifier.mat','file')
    classifier_choice = questdlg('Dataset will be freshly trained now (faceclassifier.mat file is missing from current directory). Would you like to proceed ? ', ...
        'Train CLassifier','Yes','No', 'No');
    switch classifier_choice
        case 'Yes'
            classifier_choice = 'Train freshly';
        case 'No'
            return;
    end
else
    classifier_choice = questdlg('Would you like a use already trained classifier (present as faceclassifier.mat in current directory) or wish to train the data (this may take some time) ?', ...
    'Train CLassifier','Use faceclassifier.mat','Train freshly', 'Use faceclassifier.mat');
end

if strcmp(classifier_choice, 'Use faceclassifier.mat')
    fc = load('faceClassifier.mat');
    faceTracker.faceClassifier = fc.faceClassifier;
else
    if ~exist('TrainClassifier.m','file')
        msgbox(['TrainClassifier.m file is missing from current directory. Paste it in the current folder: [' pwd ']'], 'Error','error');
        return;
    end
    TrainClassifier
    faceTracker.faceClassifier = faceClassifier;
end
    
choice = menu('Select the mode:', 'Webcam', 'Video');
switch choice
    case 1
        %Initialize webcam
        vidObj = webcam;
    case 2        
        fileIndex = 0;
        cellArrayOffiles = {};
        mp4Files = dir('*.mp4');
        aviFiles = dir('*.avi');
        files = {mp4Files.name, aviFiles.name};   
        if ~isempty(files)
            choice2 = menu('Select a file:', files);
        else
            msgbox(['No video files found in current folder. Paste them in the current folder: [' pwd ']'], 'Error','error');
            return;
        end
        videoFile = files(choice2);
        
        %Initialize video object
        vidObj = vision.VideoFileReader(char(videoFile(1)));
        
end      

showUnknownClass = questdlg('Would you like a show unknown classes in the frames ?',...
    'Classification', 'Yes', 'No', 'Yes');

if strcmp(showUnknownClass, 'No')
    faceTracker.showUnknownClass = 0;
end

%Proceed if any of the above choices are selected
if choice          
    
    %Initialize Face detector
    faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
    

    %Goto next frame
    if choice == 1
        frame = snapshot(vidObj);
    else
        frame = step(vidObj);
    end 
    
    frameSize = size(frame);    

    %Initialize video player object
    videoPlayer  = vision.VideoPlayer('Position',[150 150 frameSize(2)+50 frameSize(1)+50]);    
    
    faceDetector.MergeThreshold = 5;
    
    %Loop until a face is detected
    boxFrame = [];
    if choice ~= 1 
        while true            
            framergb = step(vidObj);                    
            frame = rgb2gray(framergb);
            boxFrame = faceDetector.step(frame);
            if ~isempty(boxFrame)
                faceTracker.classifyFaces(frame, boxFrame, framergb);
                if size(faceTracker.FBboxes,1) > 0      
                    break;
                end
            end
        end
        faceTracker.classifyFaces(frame, boxFrame, framergb);
    end
    

    %Loop until the end of the video is closed or forever in case of webcam
    frameNumber = 0;    
    while choice == 1 || ~isDone(vidObj)

        if choice == 1            
            framergb = snapshot(vidObj);
        else
            framergb = step(vidObj);
        end          
        frame = rgb2gray(framergb);

        % Redetecting faces as in 10 frames a face might go off the
        % frame or new faces might appear in it.
        % Also downsampling image to speedup the face detection call. 
        if mod(frameNumber, 20) == 0                       
            boxFrame = 2 * faceDetector.step(imresize(frame, 0.5));
            if ~isempty(boxFrame)
                faceTracker.classifyFaces(frame, boxFrame, framergb);
            end
        else
            % Track faces
            faceTracker.track(frame);
        end
        
        % Display bounding boxes if any box is still being tracked
        if size(faceTracker.FBboxes,1) > 0            
            displayFrame = insertObjectAnnotation(framergb, 'rectangle', faceTracker.FBboxes, faceTracker.personLabel);            
            videoPlayer.step(displayFrame);         
        else
            videoPlayer.step(framergb);
        end

        frameNumber = frameNumber + 1;
        
        if ~isOpen(videoPlayer)                
            break;
        end
            
    end

    % Hide and clean up
    if isOpen(videoPlayer)
        hide(videoPlayer);
    end              
    release(videoPlayer);
end
%Clear all variables, objects
clear;