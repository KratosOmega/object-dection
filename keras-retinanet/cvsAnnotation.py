######################################################################################################## imports
import glob
import json
import os
import ntpath
import csv


######################################################################################################## Methods
def load_fileNames(data_path):
    fileNames = []
    for data in glob.glob(data_path + "*.jpg"):
        fileName = ntpath.basename(data)
        nameOnly = os.path.splitext(fileName)[0]
        fileNames.append(nameOnly)
    return fileNames

def getImgs(dir_path, fileNames, h, w):
    imgs = []
    for filename in fileNames:
        img = cv2.imread(dir_path + filename + '.jpg')
        resized = cv2.resize(img,(h, w))
        imgs.append(resized)
    return np.array(imgs)

def getLabels(dir_path, fileNames):
    labels = []
    for filename in fileNames:
        with open(dir_path + filename + '.json') as file:
            label = json.load(file)
            shape_map = label['shape_type']
            char_map = label['character']
            labels.append([shape_map, char_map])
    return labels

def generateCvsAnnotation(input_path, output_path):
    imgPath = []
    for data in glob.glob(input_path + "*.jpg"):
        imgPath.append(data)

    fileNames = load_fileNames(input_path)

    labels = getLabels(input_path, fileNames)

    print(labels[0][0])
    print(labels[0][1])
    print(len(labels))
    print(len(imgPath))

    with open(output_path, mode='w') as csv_file:
        num_data = len(labels)
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, num_data):
            csv_writer.writerow([imgPath[i], '3', '3', '38', '38', labels[i][0]])
            csv_writer.writerow([imgPath[i], '15', '15', '25', '25', labels[i][1]])

    print("csv file is generated...")

def main(args=None):
    generateCvsAnnotation('../keras/cnn_training_data/', '../keras/cnn_training_data/annotation.csv')



if __name__ == '__main__':
    main()
