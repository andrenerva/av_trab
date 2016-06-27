close all;
clear all;
clc;

%Read the input image 
k = imread('imagem11.jpg');
Irecorte = imresize(k,[250 250]);
%Read the input image 
K = imread('imagem3.jpg');
Ibase = imresize(K,[250 250]);

KK = imread('branca.jpg');
branca = imresize(KK,[250 250]);

%To detect Mouth
MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',130);
BBboca=step(MouthDetect,Irecorte);
figure,
imshow(Irecorte); hold on
for i = 1:size(BBboca,1)
 rectangle('Position',BBboca(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','r');
end

%To detect nose
NoseDetect = vision.CascadeObjectDetector('Nose','MergeThreshold',10);
BBnariz=step(NoseDetect,Irecorte);
hold on
for i = 1:size(BBnariz,1)
 rectangle('Position',BBnariz(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','g');
end

%To detect Eyes
EyeDetect = vision.CascadeObjectDetector('EyePairBig');
BBolhos=step(EyeDetect,Irecorte);
hold on
rectangle('Position',BBolhos,'LineWidth',4,'LineStyle','-','EdgeColor','b');

IBoca = imcrop(Irecorte,BBboca);



INariz = imcrop(Irecorte,BBnariz);

IOlhos = imcrop(Irecorte,BBolhos);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%To detect Mouth
MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',150);
BBboca_base=step(MouthDetect,Ibase);

figure,
%imshow(Ibase); hold on
for i = 1:size(BBboca_base,1)
 %rectangle('Position',BBboca_base(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','r');
end
IBoca_base = imcrop(Ibase,BBboca_base);
%figure, imshow(IBoca_base), title('Olhos');

%To detect nose
NoseDetect = vision.CascadeObjectDetector('Nose','MergeThreshold',10);
BBnariz_base=step(NoseDetect,Ibase);
INariz_base = imcrop(Ibase,BBnariz_base);

%hold on
for i = 1:size(BBnariz,1)
 %rectangle('Position',BBnariz_base(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','g');
end
INariz_base = imcrop(Ibase,BBnariz_base);
%figure, imshow(INariz_base), title('Nariz');

%To detect Eyes
EyeDetect = vision.CascadeObjectDetector('EyePairBig');
BBolhos_base=step(EyeDetect,Ibase);

%hold on
%rectangle('Position',BBolhos_base,'LineWidth',4,'LineStyle','-','EdgeColor','b');

IOlhos_base = imcrop(Ibase,BBolhos_base);
%figure, imshow(IOlhos_base), title('Olhos');

x_base_olhos = BBolhos_base(1:1)
y_base_olhos = BBolhos_base(2:2)
x_base_nariz = BBnariz_base(1:1)
y_base_nariz = BBnariz_base(2:2)
x_base_boca = BBboca_base(1:1)
y_base_boca = BBboca_base(2:2)

x_olhos = BBolhos(1:1)
y_olhos = BBolhos(2:2)
x_nariz = BBnariz(1:1)
y_nariz = BBnariz(2:2)
x_boca = BBboca(1:1)
y_boca = BBboca(2:2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m1 n1 s1] = size(IOlhos_base)
olho = imresize(IOlhos,[m1 n1]);

[m2 n2 s2] = size(INariz_base)
nariz = imresize(INariz,[m2 n2]);

[m3 n3 s3] = size(IBoca_base)
boca = imresize(IBoca,[m3 n3]);

halphablend = vision.AlphaBlender('Opacity', 1);
local1 = vision.AlphaBlender('Location', [x_olhos y_olhos]);
local2 = vision.AlphaBlender('Location', [x_nariz y_nariz]);
local3 = vision.AlphaBlender('Location', [x_boca y_boca]);
J2 = step(local1,branca,IOlhos);
J2 = step(local2,J2,INariz);
J2 = step(local3,J2,IBoca);
J2 = step(halphablend,branca,J2);

%imshow(J2);          

BW = ones(250,250,3);

for z = 1:3
    for y = 1:250
        for x = 1:250
            if J2(x,y,z) > 200 && J2(x,y,z) <= 255
                BW(x,y,z) = 0;
            else
                BW(x,y,z) = 255;
            end
        end
    end
end

BW = im2bw(BW, 0.1);

%figure;
%imshow(BW);

CH = bwconvhull(BW, 'objects')

%figure;
%imshow(CH);

bordas = edge(CH,'sobel')
bordas = imcomplement(bordas)

m = +bordas;
M = im2uint8(bordas);

imshow(bordas)
figure;

I = bwareaopen(m, 10000, 4);
%imshow(I)
%figure;

I = repmat(+I,[1,1,3]);
%imshow(I), title('mask')
%figure;

for z = 1:3
    for y = 1:250
        for x = 1:250
            if I(x,y,z) == 1
                In(x,y,z) = 255;
            else 
                In(x,y,z) = Irecorte(x,y,z);
            end
       end 
    end
end

pixel = impixel(INariz, [10], [10]);
pixel_base = impixel(INariz_base, [15], [15]);

local = vision.AlphaBlender('Location', [x_base_olhos - x_olhos y_base_olhos - y_olhos]);
Inew = step(local, branca, In);
imshow(Inew);
figure;

[face, BB] = face_det(Inew);
[face_base, BB_base] = face_det(Ibase);

imshow(face); figure;

[altura, largura, dimensoes] = size(face_base);

face = imresize(face,[altura, largura]);
imshow(face); title('face'), figure;

new = ones(250,250,3);
posicao = vision.AlphaBlender('Location', [BB_base(1:1) BB_base(2:2)]);
face = step(posicao,branca,face);

imshow(face); title('teste');figure;

for z = 1:3
    for y = 1:250
        for x = 1:250
            if face(x,y,z) >= pixel_base(z) - 10 
                if face(x,y,z) < pixel_base(z) + 10
                    face(x,y,z) = pixel_base(z);
                end
            end 
        end
    end
end

imshow(face); title('teste pele');figure;

for z = 1:3
    for y = 1:250
        for x = 1:170
            if face(x,y,z) >= 253
               face(x,y,z) = 255;
            end
       end 
    end
end

for z = 1:3
    for y = 1:250
        for x = 1:250
            if face(x,y,z) >= 254
                In(x,y,z) = Ibase(x,y,z);
            else 
                In(x,y,z) = face(x,y,z);
            end
       end 
    end
end

%faz uma filtragem de media para retirar os ruidos do texto
H = fspecial('gaussian',[3 3],10); 
imagemFiltrada = imfilter(In,H,'replicate');
%mostra o resultado do texto sem ruidos
figure, imshow(imagemFiltrada), title('Face Swap');


figure
subplot(1,3,1);
imshow(Ibase);
subplot(1,3,2);
imshow(Irecorte);
subplot(1,3,3);
imshow(imagemFiltrada);
