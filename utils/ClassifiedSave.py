import os
# from scipy.misc import imsave
from imageio import imsave


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


filename = 'D:/论文/自监督学习/代码/test/data/cifar-10-batches-py/'  # 图片的路径
meta = unpickle(filename + '/batches.meta')
label_name = meta[b'label_names']

#将训练集中的每类图片按照类别进行分组

# for i in range(len(label_name)):
#     file = label_name[i].decode()
#     path = 'D:/论文/自监督学习/代码/test/data/cifar-10-batches-py/train/' + file
#     isExist = os.path.exists(path)
#     if not isExist:
#         os.makedirs(path)
#
#
# for i in range(1, 6):
#     content = unpickle(filename + '/data_batch_' + str(i))  # 解压后的每个data_batch_
#     print('load data...')
#     print(content.keys())
#     print('tranfering data_batch' + str(i))
#     for j in range(10000):
#         img = content[b'data'][j]
#         img = img.reshape(3, 32, 32)
#         img = img.transpose(1, 2, 0)
#         img_name = 'D:/论文/自监督学习/代码/test/data/cifar-10-batches-py/train/' + label_name[
#             content[b'labels'][j]].decode() + '/batch_' + str(i) + '_num_' + str(
#             j) + '.jpg'
#         imsave(img_name, img)
#
# for i in range(len(label_name)):
#     file = label_name[i].decode()
#     path = 'D:/论文/自监督学习/代码/test/data/cifar-10-batches-py/train/' + file
#     isExist = os.path.exists(path)
#     if not isExist:
#         os.makedirs(path)

#将测试集中的图片按照类别进行分组

for i in range(len(label_name)):
    file = label_name[i].decode()
    path = 'D:/论文/自监督学习/代码/test/data/cifar-10-batches-py/test/' + file
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

textXtr = unpickle("test_batch")
for j in range(10000):
    img = textXtr[b'data'][j]
    img = img.reshape(3, 32, 32)
    img = img.transpose(1, 2, 0)
    img_name = 'D:/论文/自监督学习/代码/test/data/cifar-10-batches-py/test/' + label_name[
        textXtr[b'labels'][j]].decode() + '/batch_' + str(i) + '_num_' + str(
        j) + '.jpg'
    imsave(img_name, img)