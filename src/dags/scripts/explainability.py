import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from tensorflow.keras.preprocessing import image
from scripts.preprocessing import load_and_preprocess_image
import warnings


#Method to create pertubations of image passed based on superprixels/segments 
def perturb_image(img,perturbation,segments):
  active_pixels = np.where(perturbation == 1)[0]
  mask = np.zeros(segments.shape)
  for active in active_pixels:
      mask[segments == active] = 1
  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask[:,:,np.newaxis]
  return perturbed_image

# Method for explainability using LIME
def explain_inference(img_array, model):
    #Xi = load_and_preprocess_image(img_path)

    Xi = np.squeeze(img_array)
    preds = model.predict(img_array) # You can also make the prdiction someplace else and take input to the method
    
    # Find the top predicted class
    class_to_explain = preds.argsort()[0, -5:][::-1][0]

     # Generate superpixels/segments 
    superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)
    num_superpixels = np.unique(superpixels).shape[0]

    # Create perturbations based on superpixels created above
    num_perturb = 100
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))


    predictions = []
    for pert in perturbations:
        perturbed_img = perturb_image(img_array,pert,superpixels)
        pred = model.predict(perturbed_img)
        predictions.append(pred)

    predictions = np.array(predictions)
    
    original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled
    distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()

    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function

    
    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
    coeff = simpler_model.coef_[0]

    num_top_features = 6
    top_features = np.argsort(coeff)[-num_top_features:]

    mask = np.zeros(num_superpixels)
    mask[top_features]= True #Activate top superpixels

    return ([perturb_image(Xi,mask,superpixels), skimage.segmentation.mark_boundaries(Xi, superpixels)])