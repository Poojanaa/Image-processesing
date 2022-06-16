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







