#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et

import os
import sys
import time
import numpy as np
import keras

import ants
import antspynet

args = sys.argv

if len(args) != 4:
    help_message = ("Usage:  python doBrainExtraction.py inputFile outputFile")
    raise AttributeError(help_message)
else:
    input_file_name = args[1]
    output_file_name = args[2]
    reorient_template_file_name = args[3]

start_time_total = time.time()

print("Reading ", input_file_name)
start_time = time.time()
image = ants.image_read(input_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("Reading reorientation template " + reorient_template_file_name)
start_time = time.time()
reorient_template = ants.image_read(reorient_template_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("Normalizing to template")
start_time = time.time()
center_of_mass_template = ants.get_center_of_mass(reorient_template)
center_of_mass_image = ants.get_center_of_mass(image)
translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
  center=np.asarray(center_of_mass_template),
  translation=translation)
warped_image = ants.apply_ants_transform_to_image(xfrm, image,
  reorient_template)
warped_image = (warped_image - warped_image.mean()) / warped_image.std()

#########################################
#
# Perform initial (stage 1) segmentation
#

print("*************  Initial stage segmentation  ***************")
# print("  (warning:  steps are somewhat different in the ")
# print("   publication.  just getting something to work)")
print("")

shape_initial_stage = (160, 160, 128)

print("    Initial step 1: bias correction.")
start_time = time.time()
image_n4 = warped_image # ants.n4_bias_field_correction(warped_image)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

# Threshold at 10th percentile of non-zero voxels in "robust range (fslmaths)"
print("    Initial step 2: threshold.")
start_time = time.time()
image_n4_array = ((image_n4.numpy()).flatten())
image_n4_nonzero = image_n4_array[(image_n4_array > 0).nonzero()]
image_robust_range = np.quantile( image_n4_nonzero, (0.02, 0.98))
threshold_value = 0.10 * (image_robust_range[1] - image_robust_range[0]) + image_robust_range[0]
thresholded_mask = ants.threshold_image(image_n4, -10000, threshold_value, 0, 1)
thresholded_image = image_n4 * thresholded_mask
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

# Standardize image (should do patch-based stuff but making a quicker substitute for testing)
print("    Initial step 3: standardize.")
start_time = time.time()
thresholded_array = ((thresholded_image.numpy()).flatten())
thresholded_nonzero = image_n4_array[(thresholded_array > 0).nonzero()]
image_mean = np.mean(thresholded_nonzero)
image_sd = np.std(thresholded_nonzero)
image_standard = (image_n4 - image_mean) / image_sd
image_standard = image_standard * thresholded_mask
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

# Resample image
print("    Initial step 4: resample to (160, 160, 128).")
start_time = time.time()
# image_resampled = ants.resample_image(image_standard, shape_initial_stage, True, 0)
image_resampled = image_standard
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

# Build model and load weights for first pass
print("    Initial step 5: load weights.")
start_time = time.time()
model_initial_stage = antspynet.create_hippmapp3r_unet_model_3d((*shape_initial_stage, 1), True)
weights_file_name = "./hippMapp3rInitialWeights.h5"

if not os.path.exists(weights_file_name):
    weights_file_name = antspynet.get_pretrained_network("hippMapp3rInitial", weights_file_name)

model_initial_stage.load_weights(weights_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

# Create initial segmentation image
print("    Initial step 6: prediction.")
start_time = time.time()
data_initial_stage = image_resampled.numpy()
data_initial_stage = np.expand_dims(data_initial_stage, 0)
data_initial_stage = np.expand_dims(data_initial_stage, -1)

prediction_initial_stage = np.squeeze(model_initial_stage.predict(data_initial_stage))
prediction_initial_stage[np.where(prediction_initial_stage >= 0.5)] = 1
prediction_initial_stage[np.where(prediction_initial_stage < 0.5)] = 0
mask_initial_stage = ants.from_numpy(prediction_initial_stage,
  origin=image_resampled.origin, spacing=image_resampled.spacing,
  direction=image_resampled.direction)
mask_initial_stage = ants.label_clusters(mask_initial_stage, min_cluster_size=10)
mask_initial_stage = ants.threshold_image(mask_initial_stage, 1, 2, 1, 0)
mask_initial_stage_original_space = ants.resample_image(mask_initial_stage, image_n4.shape, True, 1)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")


#########################################
#
# Perform initial (stage 2) segmentation
#

print("")
print("")
print("*************  Refine stage segmentation  ***************")
# print("  (warning:  These steps need closer inspection.)")
print("")

shape_refine_stage = (112, 112, 64)

# Trim image space
print("    Refine step 1: crop image centering on initial mask.")
start_time = time.time()
# centroid = np.round(ants.label_image_centroids(mask_initial_stage)['vertices'][0]).astype(int)
centroid_indices = np.where(prediction_initial_stage == 1)
centroid = list()
centroid.append(int(np.mean(centroid_indices[0])))
centroid.append(int(np.mean(centroid_indices[1])))
centroid.append(int(np.mean(centroid_indices[2])))

lower = list()
lower.append(centroid[0] - int(0.5 * shape_refine_stage[0]))
lower.append(centroid[1] - int(0.5 * shape_refine_stage[1]))
lower.append(centroid[2] - int(0.5 * shape_refine_stage[2]))
upper = list()
upper.append(lower[0] + shape_refine_stage[0])
upper.append(lower[1] + shape_refine_stage[1])
upper.append(lower[2] + shape_refine_stage[2])

mask_trimmed = ants.crop_indices(mask_initial_stage, lower, upper)
image_trimmed = ants.crop_indices(image_resampled, lower, upper)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

# Build model and load weights for second pass
print("    Refine step 2: load weights.")
start_time = time.time()
model_refine_stage = antspynet.create_hippmapp3r_unet_model_3d((*shape_refine_stage, 1), False)
weights_file_name = "./hippMapp3rRefineWeights.h5"

if not os.path.exists(weights_file_name):
    weights_file_name = antspynet.get_pretrained_network("hippMapp3rRefine", weights_file_name)

model_refine_stage.load_weights(weights_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

# Create refine segmentation image
print("    Refine step 3: do monte carlo iterations (SpatialDropout).")
start_time = time.time()
data_refine_stage = image_trimmed.numpy()
data_refine_stage = np.expand_dims(data_refine_stage, 0)
data_refine_stage = np.expand_dims(data_refine_stage, -1)

number_of_mc_iterations = 30

prediction_refine_stage = np.zeros((number_of_mc_iterations,*shape_refine_stage))
for i in range(number_of_mc_iterations):
    print("        Doing monte carlo iteration", i, "out of", number_of_mc_iterations)
    prediction_refine_stage[i,:,:,:] = np.squeeze(model_refine_stage.predict(data_refine_stage))

prediction_refine_stage = np.mean(prediction_refine_stage, axis=0)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("    Refine step 4: Average monte carlo results and write probability mask image.")
start_time = time.time()
prediction_refine_stage_array = np.zeros(image_resampled.shape)
prediction_refine_stage_array[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]] = prediction_refine_stage
probability_mask_refine_stage_resampled = ants.from_numpy(prediction_refine_stage_array,
  origin=image_resampled.origin, spacing=image_resampled.spacing,
  direction=image_resampled.direction)
probability_mask_refine_stage = ants.resample_image_to_target(
  probability_mask_refine_stage_resampled, image)
ants.image_write(probability_mask_refine_stage, output_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("Renormalize to native space")
start_time = time.time()
probability_image = ants.apply_ants_transform_to_image(
  ants.invert_ants_transform(xfrm), probability_mask_refine_stage,
  image)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("Writing", output_file_name)
start_time = time.time()
ants.image_write(probability_image, output_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

end_time_total = time.time()
elapsed_time_total = end_time_total - start_time_total
print("  (Total elapsed time: ", elapsed_time, " seconds)")
