#!/usr/bin/env python
# coding: utf-8

# In[1]:


#creating dataset
import cv2
import numpy as np
import os


# In[2]:


image_x,image_y=50,50


# In[3]:


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


# In[4]:


def store_image(g_id):
    total_pics=1200
    cap=cv2.VideoCapture(0)
    x,y,w,h=300,50,350,350
    create_folder("gestures"+str(g_id))
    pic_no=0
    flag_start_capturing=False
    frame=0
    while True:
        ret,frame=cap.read()
        frame=cv2.flip(frame,1)
        hsv=cv2.inRange(frame,cv2.COLOR_BGR2HSV)
        mask2=cv2.inRange(hsv,np.array([2,50,60]),np.array([25,150,255]))
        res=cv2.bitwise_and(frame,frame,mask=mark2)
        gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        median=cv2.GaussianBlur(gray,(5,5),0)
        
        kernel_square=np.ones((5,5),np.uint8)
        dilation=cv2.dilation(median,kernel_square,iterations=2)
        opening=cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel_square)
        
        ret,thresh=cv2.threshold(opening,30,255,cv2.THRESH_BINARY)
        thresh=thresh[y:y+h,x:x+w]
        contours=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
        
        if len(contours)>0:
            contour=max(contours,key=cv2.contourArea)
            if cv2.contourArea(contour)>10000 and frames>50:
                x1,y1,w1,h1,=cv2.boundingRect(contour)
                pic_no+=1
                save_img=thresh[y1:y1+h1,x1:x1+w1]
                save_img=cv2.resize(save_img,(image_x,image_y))
                
            cv2.putText(frame,"Capturing...",(30,60),cv2.FONT_HERSHEY_TRIPLEX,2,(127,255,255))
            cv2.imwrite("gestures"+str(g_id)+""+str(pic_no)+"jpg",save_img)
            
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,str(pic_no),(30,400),cv2.FONT_HERSHEY_TRIPLEX,1.5,(127,127,255))
        cv2.imshow("Capturing gesture",frame)
        cv2.imshow("thresh",thresh)
        keypress=cv2waitKey(1)
        if keypress==ord('c'):
            if(flag_start_capturing==False):
                flag_start_capturing=True
            else:
                flag_start_capturing=False
                frames=0
        if flag_start_capturing==True:
            frames+=1
        if pic_no==total_pics:
            break


# In[5]:


g_id=int(input("Enter gesture no.:"))
store_image(g_id)


# In[6]:


#Model
import numpy as np
from keras.layers import Dense,Flatten,Conv2D
from keras.layers import MaxPooling2D,Dropout
from keras.utils import np_utils,print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd 
import keras.backend as K 


# In[40]:


data=pd.read_csv("train_foo.csv")
dataset=np.array(data)
np.random.shuffle(dataset)
X=dataset
Y=dataset
X=X[:,0:1024]
Y=Y[:,1024]


# In[41]:


X_train=X[0:70000,:]
X_train=X_train/255.
X_test=X[70000:72001,:]
X_test=X_test/255.
Y=Y.reshape(Y.shape[0],1)


# In[42]:


Y_train=Y[0:70000,:]
Y_train=Y_train.T
Y_test=Y[70000:72001,:]
Y_test=Y_test.T


# In[43]:


print("number of training examples="+str(X_train.shape[0]))
print("number of test examples="+str(X_test.shape[0]))
print("X_train shape:"+str(X_train.shape))
print("Y_train shape:"+str(Y_train.shape))
print("X_test shape:"+str(X_test.shape))
print("Y_test shape:"+str(Y_test.shape))


# In[44]:


image_x=32
image_y=32

train_y=np_utils.to_categorical(Y_train)
test_y=np_utils.to_categorical(Y_test)
train_y=train_y.reshape(train_y.shape[0],train_y.shape[1])
test_y=test_y.reshape(test_y.shape[0],test_y.shape[1])
X_train=X_train.reshape(X_train.shape[0],image_x,image_y,1)
X_test=X_test.reshape(X_test.shape[0],image_x,image_y,1)
print("X_train shape:"+str(X_train.shape))
print("X_test shape:"+str(X_test.shape))


# In[45]:


def keras_model(image_x,image_y):
    num_of_classes=37
    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape=(image_x,image_y,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same'))
    model.add(Conv2D(64,(5,5),activation='softmax'))
    model.add(Flatten())
    model.add(Dense(num_of_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    filepath="handEmo.h5"
    checkpoint1=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callbacks_list=[checkpoint1]
    
    return model,callbacks_list


# In[46]:


model,callbacks_list=keras_model(image_x,image_y)
model.fit(X_train,train_y,validation_data=(X_test,test_y),epochs=2, batch_size=64,callbacks=callbacks_list)
scores=model.evaluate(X_test,test_y,verbose=0)
print("CNN error:%.2f%%"%(100-scores[1]*100))
print_summary(model)
model.save('handEmo.h5')


# In[47]:


import cv2
from keras.models import load_model
import numpy as np
import os
model=load_model('handEmo.h5')
def get_emojis():
    emojis_folder='handEmo/'
    emojis=[]
    for emojis in range (len(os.listdir(emojis_folder))):
        print(emojis)
        emojis.append(cv2.imread(emojis_folder+str(emojis)+'png',-1))
    return emojis


# In[48]:


def keras_predict(model,image):
    processed=keras_process_image(image)
    pred_probab=model.predict(processed)[0]
    pred_class=list(pred_probab).index(max(pred_probab))
    return max(pred_probab),pred_class


# In[49]:


def keras_process_image(img):
    image_x=50
    image_y=50
    img=cv2.resize(img,(image_x,image_y))
    img=np.array(img,dtype=np.float32)
    img=np.reshape(img,(-1,image_x,image_y,1))
    return img


# In[50]:


def overlay(image,emoji,x,y,w,h):
    emoji =cv2.resize(emoji,(w,h))
    try:
        image[y:y+h,x:x+w]=blend_transparent(image[y:y+h,x:x+w],emoji)
    except:
        pass
    return image


# In[51]:


def blend_transparent(face_img,overlay_t_img):
    overlay_img=overlay_t_img[:,:,:3]
    overlay_mask=overlay_t_img[:,:,3:]
    background_mask=255-overlay_mask
    overlay_mask=cv2.cvtColor(overlay_mask,cv2.COLOR_GRAY2BGR)
    background_mask=cv2.cvtColor(background_mask,cv2.COLOR_GRAY2BGR)
    face_part=(face_img*(1/255.0))*(background_mask*(1/255.0))
    overlay_part=(overlay_img*(1/255.0))*(overlay_mask*(1/255.0))
    return np.uint8(cv2.addWeighted(face_part,255.0,overlay_part,255.0,0.0))


# In[52]:


emojis=get_emojis
cap=cv2.VideoCapture(0)
x,y,w,h=300,50,350,350
while (cap.isOpened()):
    ret,img=cap.read()
    img=cv2.flip(img,1)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask2=cv2.inRange(hsv,np.array([2,50,60]),np.array([25,150,255]))
    res=cv2.bitwise_and(img,img,mask=mask2)
    gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    median=cv2.GaussianBlur(gray,(5,5),0)
    kernel_square=np.ones((5,5),np.uint8)
    dilation=cv2.dilate(median,kernel_square,iterations=2)
    opening=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel_square)
    ret,thresh=cv2.threshold(opening,30,255,cv2.THRESH_BINARY)
    thresh=thresh[y:y+h,x:x+w]
    contours=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
    if len (contours)>0:
        contour=max(contours,key=cv2.contourArea)
        if cv2.contourArea(contour)>2500:
            x,y,w1,h1=cv2.boundingRect(contour)
            newImage=thresh[y:y+h1,x:x+w1]
            newImage=cv2.resize(newImage,(50,50))
            pred_probab,pred_class=keras_predict(model,newImage)
            print(pred_class,pred_probab)
            img=overlay(img,emojis[pred_class],400,250,90,90)
    x,y,w,h=300,50,350,350
    cv2.imshow("Frame",img)
    cv2.imshow("Contours",thresh)
    k=cv2.waitKey(10)
    if k==27:
        break


# In[ ]:




