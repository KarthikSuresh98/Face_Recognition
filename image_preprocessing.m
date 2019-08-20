myFolder = 'data\yalefaces';
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, '*.jpg');
jpgFiles = dir(filePattern);
for w = 1:length(jpgFiles)
  baseFileName = jpgFiles(w).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  i = imread(fullFileName);
  i = double(i)/255;
  faceDetector = vision.CascadeObjectDetector;
  bbox = step(faceDetector , i);
  

  x1 = bbox(1)-15;
  y1 = bbox(2)-35;
  x2 = x1 + bbox(3)+35;
  y2 = y1 + bbox(4)+45;

  inew = i(y1:y2 , x1:x2);
  inew = imresize(inew , [224,224]);
  l = 2*log(1 + (inew));
  l = imgaussfilt(l , 0.4);  
  lnew = imsharpen(l);
  imwrite(lnew , fullFileName);
end
