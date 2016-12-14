faceDetector = vision.CascadeObjectDetector;

%Location of dataset images
location = 'myPic';

errIndex = 0;
%Fecth information of all images in dataset but don't load them in memory
imd = imageDatastore(location,'IncludeSubFolders', true, 'LabelSource', 'foldernames');
try
    %Iterate each image file entry in loaded metadata
    for i = 1:size(imd.Files, 1)
        errIndex = i;
        I = imread(char(imd.Files(i)));
        %Having default threshold value for face detection so that
        %accurately one face only is detected in the given image (assuming
        %all the images for a subject has only his face in the image not
        %anyone's else)
        defaultThreshold = 10;
        
        
        %The following logic is correct the face detection of the main
        %character in the image as faceDetector can detect multiple faces
        %even if we have only 1 face in the pic. So, in that case increase
        %the threshold value. If there is no face detected at all, decrease
        %the value so that atleast one face is detected (we assume that 
        %images in dataset has atleast character's face)
        flag = 1;
        loopLimit = 10;
        counter = 0;
        while flag && (counter < loopLimit)
            faceDetector.MergeThreshold = defaultThreshold;
            bBox = step(faceDetector, I);
            if size(bBox,1) > 1 
                defaultThreshold = defaultThreshold + 1;
            elseif size(bBox,1) == 0
                defaultThreshold = defaultThreshold - 1;
            else
                break
            end
            counter = counter + 1;
        end

        if counter >= loopLimit
            continue;
        end
        
        %Insert bounding box
        faces = insertShape(I, 'rectangle', bBox, 'LineWidth', 5);        

        boundingBox= imcrop(faces,bBox(1, :));    
        scaleFac = 150/size(boundingBox, 1);
        boundingBox = imresize(boundingBox, scaleFac);

        newFile = strrep(imd.Files(i), 'cv_imageset', 'cv2_imageset');
        newFile = char((newFile));

        newLocation = strrep(location, 'cv_imageset', 'cv2_imageset');
        if ~exist(newLocation, 'dir')
            mkdir(newLocation);
        end
        
        %Extracting subfolder name in which image needs to be stored subfolders 
        newSubfolder = strcat(newLocation, '/', char(imd.Labels(i)));

        if ~exist(newSubfolder, 'dir')
            mkdir(newSubfolder);
        end

        imwrite(boundingBox, newFile);    
    end
catch ME
    %Handle error by displaying image name which casued the error.
    char(imd.Files(errIndex))
    rethrow(ME)
end
