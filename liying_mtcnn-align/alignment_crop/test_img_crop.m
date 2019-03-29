function test_img_crop()
clc;clear
image = imread('Aaliyah_04.jpg');

crop_img = align_patch(image);  
imwrite(crop_img,'crop.png');
    	
function transImg = align_patch(img)
Coord5points = [30.2946 65.5381 48.0252 33.5493 62.7299;
                51.6963 51.5014 71.7366 92.3655 92.2041];%模板关键点
            
 key_point =[162.5656  225.1543  201.0826  170.5590  225.0684;
  160.4765  159.3038  196.9603  230.4894  227.9168  ]; %detect到的关键点

 point = double([key_point(1:5)'; key_point(6:10)']);
    tfm = cp2tform(key_point', Coord5points', 'similarity');
    imgSize = [112 96];
    transImg = imtransform(img, tfm, 'biCubic','XData', [1 imgSize(2)], 'YData', [1 imgSize(1)], 'Size', imgSize);

