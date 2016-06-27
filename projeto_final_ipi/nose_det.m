function Nose = nose_det(I)

NoseDetect = vision.CascadeObjectDetector('Nose', 'MergeThreshold', 20);

BB = step(NoseDetect,I);

figure,imshow(I);

for i = 1:size(BB,1)
    rectangle('Position',BB(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','b');
end

title('Nose Detection');

Nose = imcrop(I,BB);

figure,imshow(Nose);