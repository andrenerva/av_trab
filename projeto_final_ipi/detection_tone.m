function I = detection_tone(img)

img = rgb2ycbcr(img);

for i=1:size(img,1)
    for j= 1:size(img,2)
        cb = img(i,j,2);
        cr = img(i,j,3);
        if(~(cr > 132 && cr < 173 && cb > 76 && cb < 126))
            img(i,j,1)=235;
            img(i,j,2)=128;
            img(i,j,3)=128;
        end
    end
end
img=ycbcr2rgb(img);
figure, imshow(img);