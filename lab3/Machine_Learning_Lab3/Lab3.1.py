from PIL import Image
import numpy as np
import os
from feature import NPDFeature as Fea
import pickle
from ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

print('是否存在保存好的特征文件？(1/0，如果没有，则耗费长时间重新提取...)')
t=input('请输入：')
if int(t)==0:
    print('-------开始提取特征-------')
    face_path='datasets/original/face'#文件夹路径
    nonface_path='datasets/original/nonface'
    face_dir=os.listdir(face_path)#文件夹路径中的文件名目录
    nonface_dir=os.listdir(nonface_path)

    #读取files_path+files_dir[i]的文件，使用PIL中的Image读取图片,转化为转为形状为[m,n_H,n_W]的numpy数组[数量，图片高度，图片宽度]
    def load_images(files_dir,files_path):
        datas = []
        for file_name in files_dir:
            file_path = os.path.join(files_path, file_name)  # 把文件夹和文件地址合并
            file = Image.open(file_path).convert("L")  # 读取图片，灰度化-L模式
            file = file.resize((24, 24))  # 把图片转为24x24的图片
            datas.append(np.array(file))
            file.close()
        datas = np.array(datas)#把list变为array
        return datas

    #载入数据
    face_datas=load_images(face_dir,face_path)
    nonface_datas=load_images(nonface_dir,nonface_path)

    #定义一个提取特征，返回特征数组[num,feature]的函数
    def extract_faeture(datasets,feature_extractor):
        data_features=[]
        for data in datasets:
            feature=feature_extractor(data).extract()
            data_features.append(feature)
        data_features=np.array(data_features)
        return data_features

    #调用函数得到特征值
    face_data_features=extract_faeture(face_datas,Fea)
    nonface_data_features=extract_faeture(nonface_datas,Fea)
    print('-------特征提取完毕-------')

    #使用pickle的dump()方法保存特征值到文件，下次直接load，不用上述的提取特征操作
    face_data_features_file_save = open("face_data_features.pkl", "wb")
    pickle.dump(face_data_features,face_data_features_file_save)
    nonface_data_features_file_save = open("nonface_data_features.pkl", "wb")
    pickle.dump(nonface_data_features,nonface_data_features_file_save)
else:
    print('-------加载特征文件-------')
    #使用pickle的load()方法加载特征值
    face_data_features_file_read=open("face_data_features.pkl", "rb")
    face_data_features=pickle.load(face_data_features_file_read)
    nonface_data_features_file_read=open("nonface_data_features.pkl", "rb")
    nonface_data_features=pickle.load(nonface_data_features_file_read)
    print('-------加载完毕-------')

#分别给face和nonface的数据添加label
facelabel=np.full((len(face_data_features),1),1)#创建一列全为1的数组
nonfacelabel=np.full((len(nonface_data_features),1),-1)
face_data_features=np.column_stack((face_data_features,facelabel))#把label和feature合并为两列
nonface_data_features=np.column_stack((nonface_data_features,nonfacelabel))

#把两种label数据合并再随机打乱
datasets=np.row_stack((face_data_features,nonface_data_features))
np.random.shuffle(datasets)

#切分数据集7：3训练集和验证集,并把label和data分开
length=len(datasets)//10
train_data=datasets[ :7*length]
train_X, train_y = np.hsplit(train_data,[train_data.shape[1]-1])
dev_data=datasets[7*length:]
dev_X, dev_y = np.hsplit(dev_data,[dev_data.shape[1]-1])

#调用ensemble种的adaboost模型
model=AdaBoostClassifier(40)#设置40个基分类器
model.fit(train_X,train_y)#进行训练
pre_score=model.predict_scores(train_X)
pre=model.Predict(dev_X)

dev_y=np.squeeze(dev_y)
acc=1-(np.sum(np.abs(dev_y-pre))/2)/len(dev_y)
print("Acc_value:{:.2f}%".format(acc*100))

#使用sklearn.metrics库的classification_report()函数将预测结果写入classifier_report.txt中
target_names = ['face', 'nonface']
x=classification_report(dev_y,pre,target_names=target_names)
f = open('classifier_report.txt', 'w')
f.write(x)
f.close()
