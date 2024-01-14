import joblib


model = joblib.load('main.joblib')



predictions = model.predict("New_data")

print(predictions)