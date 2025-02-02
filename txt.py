import pickle
with open('xgb_model2.pkl', 'rb') as f:
    model = pickle.load(f)
    
model.save_model('xgb_model3.json')  # Use JSON for better compatibility2

