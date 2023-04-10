from flask import Flask,url_for,render_template,request
import os
import cv2
import tensorflow as tf
app=Flask(__name__)


UPLOAD_FOLDER='E:\Major\CHEST_NET_-DENSET121-CUSTOM-\static'

categories=['FIT_LUNGS','SICK_LUNGS']
model = tf.keras.models.load_model('CHEST_NET_121_CNN.h5')

def predict(filepath):
    img_size=110
    img_array=cv2.imread(filepath)
    img_array=cv2.resize(img_array,(img_size,img_size))
    new_array=img_array.reshape(-1,img_size,img_size,3) #[BATCHSIZE,*DIMENSIONS*,COLOR_CHANNELS]
    new_array=new_array/255.0
    prediction=model.predict([new_array])
    if prediction[0][0] > 0.5:
        return categories[1]
    elif prediction[0][0] <= 0.5:
        return categories[0]
    
@app.route("/",methods=['GET','POST'])

def upload_predict():
    if request.method=='POST':
        image_file=request.files['image']
        if image_file:
            image_location=os.path.join(UPLOAD_FOLDER,image_file.filename)
            image_file.save(image_location)
            pred=predict(image_location)
            return render_template('index.html',Prediction=pred,image=image_file.filename)
        else:
            return render_template('index.html',Prediction='Upload an image',image=None)
            
    else:
        return render_template('index.html',Prediction='Upload an image',image=None)

if __name__=='__main__':
    app.run(debug=True)
