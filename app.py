from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)

model=pickle.load(open('XGB_model.pkl','rb'))
house=pd.read_csv('Cleaned_House_data.csv')

@app.route('/',methods=['GET','POST'])
def index():
    bedrooms=sorted(house['bedrooms'].unique())
    bathrooms=sorted(house['bathrooms'].unique())
    grades=sorted(house['grade'].unique())
    floors=sorted(house['floors'].unique())
    waterfronts=sorted(house['waterfront'].unique())
    views=sorted(house['view'].unique())
    conditions=sorted(house['condition'].unique())
    yr_builts=sorted(house['yr_built'].unique())
    yr_renovateds=sorted(house['yr_renovated'].unique())
    lat=house['lat'].unique()
    long=house['long'].unique()
    sqft_basement=house['sqft_basement'].unique()
    sqft_living=house['sqft_living'].unique()
    sqft_lot=house['sqft_lot'].unique()
    sqft_living15=house['sqft_living15'].unique()
    sqft_lot15=house['sqft_lot15'].unique()



    
    
    

    #bedrooms.insert(0,'Select bedroom')
    return render_template('index.html',bedrooms=bedrooms, bathrooms=bathrooms, floors=floors
                           ,grades=grades,sqft_living=sqft_living,sqft_lot=sqft_lot,
                           waterfronts=waterfronts,views=views
                           ,conditions=conditions,sqft_basement=sqft_basement,yr_builts=yr_builts,yr_renovateds=yr_renovateds,
                           lat=lat,long=long,sqft_living15=sqft_living15,sqft_lot15=sqft_lot15)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    bedroom=request.form.get('bedroom')
    bathroom=request.form.get('bathroom')
    sqft_living=request.form.get('sqft_living')
    sqft_lot=request.form.get('sqft_lot')
    floors=request.form.get('floors')
    waterfront=request.form.get('waterfront')
    view=request.form.get('view')
    condition=request.form.get('condition')
    grade=request.form.get('grade')
    sqft_above=request.form.get('sqft_above')
    sqft_basement=request.form.get('sqft_basement')
    yr_built=request.form.get('yr_built')
    yr_renovated=request.form.get('yr_renovated')
    lat=request.form.get('lat')
    long=request.form.get('long')
    sqft_living15=request.form.get('sqft_living15')
    sqft_lot15=request.form.get('sqft_lot15')


    prediction=model.predict(pd.DataFrame(columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
       'sqft_living15', 'sqft_lot15'],data=np.array([ 8,  1,  5748,  1457,
          5.0,  1,  1,  2,
          5,  12865,  0 , 1955,
          0,  99.5112, -180.258,  1340,
          5650]).reshape(1,17)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run()