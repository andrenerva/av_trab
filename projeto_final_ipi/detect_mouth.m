function I = detect_mouth(Irecorte)

%To detect Mouth
MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',130);
BBboca=step(MouthDetect,Irecorte);
figure,
imshow(Irecorte); hold on
for i = 1:size(BBboca,1)
 rectangle('Position',BBboca(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','r');
end
