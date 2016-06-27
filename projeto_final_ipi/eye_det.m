function Eyes = eye_det(I)

EyeDetect = vision.CascadeObjectDetector('EyePairBig', 'MergeThreshold', 16);

BB=step(EyeDetect,I);

figure,imshow(I);

for i = 1:size(BB,1)
    rectangle('Position',BB(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','b');
end

title('Eyes Detection');

Eyes = imcrop(I,BB);

figure,imshow(Eyes);