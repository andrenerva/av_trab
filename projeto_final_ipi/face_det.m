function [J, BB] = face_det(I)

%To detect Face
FDetect = vision.CascadeObjectDetector;

%Returns Bounding Box values based on number of objects
BB = step(FDetect,I);

J = imcrop(I,BB);