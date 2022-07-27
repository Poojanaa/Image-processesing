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
![image](https://user-images.githubusercontent.com/98141711/175275062-20b6b363-555d-4cb9-a11a-f8a556a1cdd3.png)<br>
![image](https://user-images.githubusercontent.com/98141711/175275111-ac91cd47-6938-4cb1-a34b-caccf14f5635.png)<br>
![image](https://user-images.githubusercontent.com/98141711/175275150-ed0523f7-2697-4c4c-b3a1-9865f95f57bc.png)<br>
![image](https://user-images.githubusercontent.com/98141711/175275219-7b0cd1af-915d-4ff8-96b5-7898ef2eeaae.png)<br>
**14.Program  to create an image using 2D array.**<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,:100]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('image1.png')<br>
img.show()<br>
c.waitKey(0)<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/175276586-ca3e5d9f-06e8-4cbe-a85d-fa2a195ffa30.png)<br>
<br>
**15.Program to create and image using bitwise opertaions.**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('image1.jpg')<br>
image2=cv2.imread('image1.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr= cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/176403907-f51e33c7-f921-4de5-9a1e-3d1057f6a035.png)<br>
<br>
<br>
**16.Program to blue the image.**<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('image1.jpg')<br>
cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>
Gassian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gassian)<br>
cv2.waitKey(0)<br>
median=cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/176412032-59ed2cd5-24cd-4c2b-b276-28f290e3a1c8.png)<br>
![image](https://user-images.githubusercontent.com/98141711/176412218-e34ff6b7-5228-418b-957a-78e599594023.png)<br>
![image](https://user-images.githubusercontent.com/98141711/176413116-81bea727-5e29-4294-b270-f0cc72af6d26.png)<br>
![image](https://user-images.githubusercontent.com/98141711/176413225-535332c8-3d9d-4f6e-b512-6c225964a7e4.png)<br>
<br>
**17.program to create an image enchancement.**<br>
from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('image1.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/176418759-32c027b8-f293-4442-834f-ee206c9ae8ee.png)<br>
![image](https://user-images.githubusercontent.com/98141711/176418847-d4d3d749-1a0e-40a0-9a41-16a7292a69f1.png)<br>
![image](https://user-images.githubusercontent.com/98141711/176418929-114ba10d-b072-409d-a416-72d188b5c5b9.png)<br>
![image](https://user-images.githubusercontent.com/98141711/176419007-a0e1d689-6a56-4646-a0f3-c36abb9b4b16.png)<br>
<br>
**18.Program to create a image morphology.**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('image1.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/176423771-442770eb-5da3-489f-9a32-1e49bee5ce0c.png)<br>
<br>
**19. Develop a program to**<br>
(1) Read the image,<br>
(ii) write (save) the grayscale image and<br>
(iii) display the original image and grayscale image<br>
(Note: To save image to local storage using Python, we use cv2.imwrite() function on<br>
OpenCV library)<br>
import cv2<br>
OriginalImg=cv2.imread('image1.jpg')<br>
GrayImg=cv2.imread('image1.jpg',0)<br>
isSaved=cv2.imwrite('D:\image1.jpg', GrayImg)<br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image', GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:print('The image is successfully saved.')<br>
**OUTPUT:-**<br>
The image is successfully saved.<br>
![image](https://user-images.githubusercontent.com/98141711/178700548-5184d941-bad8-430f-8ccd-4e815be9db23.png)<br>
![image](https://user-images.githubusercontent.com/98141711/178701129-12e15d82-118c-4593-bff7-7408ea9095bb.png)<br>
<br>
**20.DEVELOP A PROGRAM TO Graylevel slicing with background**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('image1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/178706628-3ae4d6ea-27ec-4c69-bfa9-c6693a67efb2.png)<br>
<br>
**21.DEVELOP A PROGRAM TO Graylevel slicing without background**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('image1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing w/o backgraound')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/178710213-195b5492-86a5-40f6-a33b-7be82ca01c2a.png)<br>
<br>
**22. DEVELOP A PROGRAM TO HISTOGRAM THE IMAGE USING SKIPY**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('moon.jpeg')<br>
_ = plt.hist(image.ravel(), bins = 256, color = 'orange', )<br>
_ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count')<br>
_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])<br>
plt.show()<br>
<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/178953279-bf2e3f77-1309-43e3-9906-2d61a236b215.png)<br>
<br>
<br>
**23. DEVELOP A PROGRAM TO HISTOGRAM THE IMAGE USING open cv2.**<br>
import cv2 <br>
from matplotlib import pyplot as plt  <br>
img = cv2.imread('moon.jpeg',0) <br>
histr = cv2.calcHist([img],[0],None,[256],[0,256]) <br>
plt.plot(histr) <br>
plt.show()<br>
<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/178956801-c9ba73d7-3c40-4983-bdfd-fb4366397acc.png)<br>
<br>
**24.DEVELOP A PROGRAM TO HISTOGRAM THE IMAGE USING NUMPY.**<br>
import numpy as np<br>
import cv2 as cv<br>
from matplotlib import pyplot as plt<br>
img = cv.imread('moon.jpeg')<br>
plt.imshow(img)<br>
plt.show()<br>
img = cv.imread('moon.jpeg',0)<br>
plt.hist(img.ravel(),256,[0,256]);<br>
plt.show()<br>
<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/178958007-611122d8-f419-4af9-bf07-ff7d8a0babbf.png)<br>
<br>
<br>
**25.DEVELOP A PROGRAM TO HISTOGRAM THE IMAGE USING PILLOW.**<br>
from PIL import Image
 img = Image.open(r"C:\Users\ADMIN\Desktop\1st Msc\images\moon.jpeg")<br>
r, g, b = img.split()<br>
len(r.histogram())<br>
r.histogram()<br>
<br>
**OUTPUT:-**<br>
![image](https://user-images.githubusercontent.com/98141711/178965513-b0749b39-3c84-47d8-a492-c49e43fd1b06.png)<br>
<br>
<br>
**26.Program to perform basic image data analysis using intensity transformation.**<br>
**a) Image negative<br>**
**b)Log transformation<br>**
**c)Gramma correction<br>**
a)<br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('bird.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180174535-49d4d081-104c-404e-b83a-601e483f4d8b.png)<br>
<br>
b)<br>
negative=255- pic # neg = (L-1) - img<br>
plt.figure(figsize= (6,6))<br>
plt.imshow(negative); <br>
plt.axis('off');<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180175209-7a5fe07d-283d-46bf-bfcd-fe368376a0c3.png)<br>
<br>
c)<br>
%matplotlib inline<br>
import imageio<br>
import numpy as np<br> 
import matplotlib.pyplot as plt<br>
pic=imageio.imread("bird.jpg")<br>
gray=lambda rgb : np.dot(rgb[...,:3], [0.299,0.587,0.114]) <br>
gray=gray(pic)<br>
max_=np.max(gray)<br>
def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(), cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180180867-15b75d7a-24b7-43bf-bba9-a950a2d4174a.png)<br>
<br>
d)<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
# Gamma encoding<br>
pic=imageio.imread('bird.jpg') <br>
gamma=2.2# Gamma < 1 Dark; Gamma > 1~ Bright<br>
gamma_correction=((pic/255)**(1/gamma)) <br>
plt.figure(figsize=(5,5)) <br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180181653-6fcfa572-adce-4fd6-850a-7e96b0011d05.png)<br>
<br>
**27.Program to perform basic image manipulation**<br>
**a)Sharpness**<br>
**b)Flipping**<br>
**c)Cropping**<br>
a)<br>
# Image sharpen<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
# Load the image<br>
my_image=Image.open('teddy.jpg')<br>
# Use sharpen function<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
# Save the image<br>
sharp.save('D:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180184934-a5cb7f63-639f-479d-8879-5a1e25663e62.png)<br>
![image](https://user-images.githubusercontent.com/98141711/180184477-1229f701-0c28-4e94-8aa3-99ab5bdcac07.png)<br>
<br>
b)<br>
import matplotlib.pyplot as plt<br>
# Load the image<br>
img =Image.open('teddy.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
# use the flip function<br>
flip=img.transpose (Image.FLIP_LEFT_RIGHT)<br>
# save the image<br>
flip.save('D:/image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180186015-ce11f704-9a6f-456c-b963-8b78e2c0059a.png)<br>
<br>
c)<br>
# Importing Image class <br>
from PIL import Image<br>
import matplotlib.pyplot as plt <br>
# Opens a image in RGB mode<br>
im = Image.open('teddy.jpg')<br>
#Size of the image in pixels (size of original image)<br>
#(This is not mandatory) <br>
width, height = im.size<br>
# Cropped image of above dimension<br>
# (It will not change original image) <br>
im1= im.crop ((280,100, 800, 600))<br>
# Shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180189374-263fcb31-411e-4c97-8118-2da1f99d7c71.png)<br>
<br>
**28.Generate matrix and display the image data.**<br>
import matplotlib.image as image<br>
img=image.imread('teddy.jpg')<br>
print('The Shape of the image is:',img.shape)<br>
print('The image as array is:')<br>
print(img)<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180191181-cc3ac009-6deb-411f-8a0f-bb155abd9c42.png)<br>
![image](https://user-images.githubusercontent.com/98141711/180191269-fcc1e6c7-f8ed-4d57-aa95-c293c1393974.png)<br>
<br>
**29.program to find the brightness of a image from distance to center.**<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (255, 255, 255)<br>
outerColor = (0,0,0)<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        arr[y, x] = (int(r), int(g), int(b))<br>
        plt.imshow(arr, cmap='gray')<br>
        plt.show()<br>
 <br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180199036-65a6fa89-d817-4295-b4e7-f158ec5f7311.png)<br>
<br>
**30.program to display the different color in diagonal with matrix.**<br>
from PIL import Image<br>
import numpy as np<br>
w, h = 512, 512<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [255, 0, 255]<br>
data[200:300, 200:300] = [0, 255, 0]<br>
# red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
plt.imshow(img)<br>
plt.show()<br>
<br>
**Output:-**<br>
![image](https://user-images.githubusercontent.com/98141711/180200493-6c5de916-8cf8-42bf-8827-2cf129b93945.png)<br>
<br>
<br>
<br>
<br>
from PIL import Image<br>
import numpy as np<br>
w, h = 600, 600<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [255, 0, 255]<br>
data[200:300, 200:300] = [0, 255, 0]<br>
data[300:400,300:400]=[0,0,255]<br>
data[400:500,400:500]=[255,255,0]<br>
data[500:600,500:600]=[0,255,255]<br>
# red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
plt.imshow(img)<br>
plt.show()<br>
<br>
**31.Read an image to find max,min,average and standard deviation of pixel value.**<br>


 
       



















    












































