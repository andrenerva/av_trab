function Mouth = mouth_det(I)

MouthDetect = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', 128);

BB = step(MouthDetect,I);

figure,imshow(I);

for i = 1:size(BB,1)
    rectangle('Position',BB(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','b');
end

title('Mouth Detection');

Mouth = imcrop(I,BB);

figure,imshow(Mouth);