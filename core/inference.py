import numpy as np
from core.preprocessing import custom_preprocessing

def predict_stroke(model, image, model_name="ResNet50"):
    # 1. Preprocess specific to the model
    img_processed = custom_preprocessing(image, model_name)
    
    # 2. Add batch dimension
    img_batch = np.expand_dims(img_processed, axis=0)
    
    # 3. Predict
    pred = model.predict(img_batch, verbose=0)[0]
    
    # Handle list vs scalar output
    if isinstance(pred, (list, np.ndarray)) and len(pred) > 0:
        score = pred[0]
    else:
        score = pred
        
    return score, score > 0.5