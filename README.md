# Image-processesing<br>
**1.Develop a program to display Grayscale image using read and write operations.**<br>
import cv2<br>
img=cv2.imread('b3.png',0)<br>
cv2.imshow('b3',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/173803534-5feb84c6-e811-4f77-8db1-3e4670580d3d.png)<br>
<br>
**2.Develop a program to display image using matPlot.lib**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image=cv2.imread('b3.png')<br>
plt.imshow(image)<br>
plt.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/173805933-18769988-cdb6-4c24-9840-1e8ce569f6d0.png)<br>
<br>
**3.Develop a program to perform linear transformation rotation.**<br>
from PIL import Image<br>
Original_Image=Image.open('b3.png')<br>
rotate_image1=Original_Image.rotate(180)<br>
rotate_image1.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/173810563-fce9731b-6c20-4ddf-a8a1-2f35b2f1a15f.png)<br>
<br>
**4.Develop a program to convert color string to RGB color values.**<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
**Output:-**<br>
(255, 255, 0)<br>
(255, 0, 0)<br>
<br>
**5.Write a program image using colors.**
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,0,0))<br>
img.show()<br>
**Outpuit:-**<br>
![image](https://user-images.githubusercontent.com/98141711/173810740-daca2918-89ff-479b-8b7a-acec028416d7.png)<br>
<br>
**6.Develop a program to initalize the imge using various color.**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('b3.png')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/173816748-feaf3062-2953-4436-b427-a799029fc0d3.png)<br>
![image](https://user-images.githubusercontent.com/98141711/173816823-e0ec35cb-b247-45e6-8330-5f08e64de099.png)<br>
**7.Write a program to display the image attributes.**<br>
from PIL import Image<br>
image=Image.open('b3.png')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
**Output:-**<br>
Filename: b3.png<br>
Format: PNG<br>
Mode: RGB<br>
Size: (840, 583)<br>
Width: 840<br>
Height: 583<br>
<br>
**8.Resize the original image.**<br>
import cv2<br>
img=cv2.imread('b3.png')<br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>
**Outputy:-**<br>
original image length width (583, 840, 3)<br>
![image](https://user-images.githubusercontent.com/98141711/174057743-dd1d4d44-34aa-4b68-8a82-4097709b6b09.png)<br>
Resized image length width (160, 150, 3)<br>
![image](https://user-images.githubusercontent.com/98141711/174057963-7c0159b6-7783-4f2c-aa76-fa5fb770e0f5.png)<br>
<br>
**9.Convert the original to grey scale and then to binary import cv2.**<br>
import cv2 <br>
img=cv2.imread('b3.png')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
img=cv2.imread('b3.png',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/174048843-e4d1eb9c-9b75-4fb5-9e24-066ef855b7fa.png)<br>
![image](https://user-images.githubusercontent.com/98141711/174049224-7db64b2a-9ab9-4a82-8c81-f9ee5dc06d6e.png)<br>
![image](https://user-images.githubusercontent.com/98141711/174049512-edb8ca08-503d-4e4f-bd32-c0cb98f6311f.png)<br>
<br>
**10.Develop a program to readimage using URL.**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://d1whtlypfis84e.cloudfront.net/guides/wp-content/uploads/2019/08/03091106/Trees-1024x682.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/175008117-11bb6bfa-8898-4ffe-86c0-364c2c504ff7.png)<br>
**11.Write a program to mask and blur the image.**<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread('flowers5.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/175018126-a6f4a12e-e90f-4d5b-a457-def12b366120.png)<br>
<br>
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1, 190, 200)<br>
dark_orange=(18, 255, 255)<br>
mask=cv2.inRange(hsv_img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/175258334-f672a6ef-8387-4138-ade8-409a036baf30.png)<br>
<br>
light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/175259588-fabf8a05-894e-4713-9d3d-ebc5596d404c.png)<br>
<br>
final_mask=mask + mask_white<br>
final_result = cv2.bitwise_and (img, img, mask=final_mask)<br>
plt.subplot(1, 2, 1)<br>
plt.imshow(final_mask, cmap="gray")<br>
plt.subplot(1, 2, 2)<br>
plt.imshow(final_result)<br>
plt.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/175260067-e65ce218-cff3-4d8d-91eb-0a7722a758e6.png)<br>
<br>
blur=cv2.GaussianBlur(final_result,(7, 7), 0)<br>
plt.imshow(blur)<br>
plt.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/175260401-2dfe2cd1-727f-4229-8079-94689db11520.png)<br>
<br>
**12.Write a program to perform arithmatic operations on images.**<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img1=cv2.imread('leaf3.jpg')<br>
img2=cv2.imread('leaf6.jpg')<br>
fimg1=img1+img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4=img1/img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg4)<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141711/175270013-2fa145b6-b5ca-44e1-830e-d6ebedc02faf.png)<br>
![image](https://user-images.githubusercontent.com/98141711/175270236-6f5c7ee9-dd44-4478-b394-3525654436b6.png)<br>
**13.Develop the program to change the image in different color spaces.**<br>
import cv2 <br>
img=cv2.imread("plants4.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/175274879-692affe2-af1d-4084-aa19-707d646feb93.png)<br>



**14**















