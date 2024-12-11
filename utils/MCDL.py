import os
# from keras.models import *
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from scipy import io
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import segyio
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
inline=601
xline=951
time=288
with segyio.open(r'C:\Users\hjl15\Desktop\PostGraduate\HorizonPicking\F3_seismic_crop.segy',"r+",ignore_geometry = True) as segyfile:
    gx=np.zeros(shape=(inline*xline,time))
    for i in range(inline*xline):
        gx[i,:]=segyfile.trace[i]
    gx=np.reshape(gx,newshape=(inline,xline,time))#分别对应三维数据体的(inline，xline，time)
    # gx = gx[64:416,128:768,64:320]
    gx = np.array(gx)
    print(gx.shape)
    seist = np.transpose(gx,(1,2,0))        #(1000, 150, 450)#################cell名称（matlab第三维度表示切片数）
    seist2 = np.transpose(gx,(0,2,1))

a=seist.shape[1]-seist.shape[1]%2
b=seist.shape[2]-seist.shape[2]%2

seist=seist[:,0:a,0:b]   #第一维是切片个数，后两维代表切片（后两维需是2的倍数）
np.save('seist.npy', seist)

print(seist.shape,'****')
# from oneoutcnn_res import hunet
# model=hunet()
# model=load_model("check/oneout/100.hdf5")  #直接cnn
# model=load_model("model_3_21.hdf5") ####集成5传统

a = np.mean(seist)
b = np.std(seist)
seist = seist - a
seist = seist / b
print(seist.shape)

# gx= model.predict(seist, verbose=1,batch_size=1)
# lablet=np.array(gx)
# print(lablet.shape)


###################################################################
a=seist2.shape[1]-seist2.shape[1]%2
b=seist2.shape[2]-seist2.shape[2]%2
seist2=seist2[:,0:a,0:b]   #第一维是切片个数，后两维代表切片（后两维需是2的倍数）

print(seist2.shape)
a = np.mean(seist2)
b = np.std(seist2)
seist2 = seist2 - a
seist2 = seist2 / b

# #
# # gx2= model.predict(seist2, verbose=1,batch_size=1)
# # lablet2=np.array(gx2)
# # print(lablet2.shape)
# #
# #
#
# io.savemat('5result_7_4.mat', {
#                                            'predict_semb_crossline':np.transpose(lablet[2,:,:,:,0]),
#                                            'predict_MCDL_crossline':np.transpose(lablet[4,:,:,:,0]),
#
#
#                                            'predict_semb_inline':np.transpose(lablet2[2,:,:,:,0]),
#                                            'predict_MCDL_inline':np.transpose(lablet2[4,:,:,:,0])
#
#

                                        # })