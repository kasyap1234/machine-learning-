dir=""
categories=['dogs','cats']
data=[]
for category in categories : 
    folder=os.path.join(dir,category)
    label=categories.index(category)
    for image in os.listdir(folder): 
        image_path=os.path.join(folder,image)
        image_=cv2.imread(image_path)
        image__=cv2.resize(image,(220,220))
        data.append([image__,label])
        plt.show(image__)
