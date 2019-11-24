library( ANTsR )
library( ANTsRNet )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) != 3 )
  {
  helpMessage <- paste0( "Usage:  Rscript doHippMapp3RSegmentation.R inputFile outputFile reorientationTemplate\n" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  outputFileName <- args [2]
  reorientTemplateFileName <- args[3]
  }

startTimeTotal <- Sys.time()

cat( "Reading ", inputFileName, "\n" )
startTime <- Sys.time()
image <- antsImageRead( inputFileName, dimension = 3 )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Reading reorientation template", reorientTemplateFileName, "\n" )
startTime <- Sys.time()
reorientTemplate <- antsImageRead( reorientTemplateFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Normalizing to template\n" )
startTime <- Sys.time()
centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
centerOfMassImage <- getCenterOfMass( image )
xfrm <- createAntsrTransform( type = "Euler3DTransform",
  center = centerOfMassTemplate,
  translation = centerOfMassImage - centerOfMassTemplate )
warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n\n" )

#########################################
#
# Perform initial (stage 1) segmentation
#

cat( "*************  Initial stage segmentation  ***************\n" )
# cat( "  (warning:  steps are somewhat different in the \n" )
# cat( "   publication.)\n" )
cat( "\n" )

shapeInitialStage <- c( 160, 160, 128 )

cat( "    Initial step 1: bias correction.\n" )
startTime <- Sys.time()
imageN4 <- n4BiasFieldCorrection( warpedImage )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Threshold at 10th percentile of non-zero voxels in "robust range (fslmaths)"
cat( "    Initial step 2: threshold.\n" )
startTime <- Sys.time()
imageArray <- as.array( imageN4 )
imageRobustRange <- quantile( imageArray[which( imageArray != 0 )], probs = c( 0.02, 0.98 ) )
thresholdValue <- 0.10 * ( imageRobustRange[2] - imageRobustRange[1] ) + imageRobustRange[1]
thresholdedMask <- thresholdImage( imageN4, -10000, thresholdValue, 0, 1 )
thresholdedImage <- imageN4 * thresholdedMask
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Standardize image
cat( "    Initial step 3: standardize." )
startTime <- Sys.time()
meanImage <- mean( thresholdedImage[thresholdedMask == 1] )
sdImage <- sd( thresholdedImage[thresholdedMask == 1] )
imageNormalized <- ( imageN4 - meanImage ) / sdImage
imageNormalized <- imageNormalized * thresholdedMask
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Resample image
cat( "    Initial step 4: resample to (160, 160, 128).\n" )
startTime <- Sys.time()
# imageResampled <- resampleImage( imageNormalized, shapeInitialStage,
#   useVoxels = TRUE, interpType = "linear" )
imageResampled <- imageNormalized
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "    Initial step 5: Load weights.\n" )
startTime <- Sys.time()
modelInitialStage <- createHippMapp3rUnetModel3D( c( shapeInitialStage, 1 ), doFirstNetwork = TRUE )
weightsFileName <- paste0( getwd(), "/hippMapp3rInitialWeights.h5" )
if( ! file.exists( weightsFileName ) )
  {
  weightsFileName <- getPretrainedNetwork( "hippMapp3rInitial", weightsFileName )
  }
modelInitialStage$load_weights( weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "    Initial step 6: prediction.\n" )
startTime <- Sys.time()
dataInitialStage <- array( data = as.array( imageResampled ), dim = c( 1, dim( imageResampled ), 1 ) )
maskArray <- modelInitialStage$predict( dataInitialStage )
maskImageResampled <- as.antsImage( maskArray[1,,,,1] ) %>% antsCopyImageInfo2( imageResampled )
maskImage <- resampleImage( maskImageResampled, dim( image ), useVoxels = TRUE,
  interpType = "nearestNeighbor" )
maskImage[maskImage >= 0.5] <- 1
maskImage[maskImage < 0.5] <- 0
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

#########################################
#
# Perform initial (stage 2) segmentation
#

cat( "\n" )
cat( "\n" )
cat( "*************  Refine stage segmentation  ***************\n" )
# cat( "  (warning:  These steps need closer inspection.)\n" )
cat( "\n" )

shapeRefineStage <- c( 112, 112, 64 )

cat( "    Refine step 1: crop to the initial estimate.\n" )
maskArray <- drop( maskArray )
centroidIndices <- which( maskArray == 1, arr.ind = TRUE, useNames = FALSE )
centroid <- rep( 0, 3 )
centroid[1] <- mean( centroidIndices[, 1] )
centroid[2] <- mean( centroidIndices[, 2] )
centroid[3] <- mean( centroidIndices[, 3] )
lower <- floor( centroid - 0.5 * shapeRefineStage )
upper <- lower + shapeRefineStage - 1
maskTrimmed <- cropIndices( maskImageResampled, lower, upper )
imageTrimmed <- cropIndices( imageResampled, lower, upper )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "    Refine step 2: generate second network and download weights.\n" )
startTime <- Sys.time()
modelRefineStage <- createHippMapp3rUnetModel3D( c( shapeRefineStage, 1 ), FALSE )
weightsFileName <- paste0( getwd(), "/hippMapp3rRefineWeights.h5" )
if( ! file.exists( weightsFileName ) )
  {
  weightsFileName <- getPretrainedNetwork( "hippMapp3rRefine", weightsFileName )
  }
modelRefineStage$load_weights( weightsFileName )
dataRefineStage <- array( data = as.array( imageTrimmed ), dim = c( 1, shapeRefineStage, 1 ) )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "    Refine step 3: do monte carlo iterations (SpatialDropout).\n" )
startTime <- Sys.time()
numberOfMCIterations <- 30
predictionRefineStage <- array( data = 0, dim = c( numberOfMCIterations, shapeRefineStage ) )
for( i in seq_len( numberOfMCIterations ) )
  {
  cat( "        Doing monte carlo iteration", i, "out of", numberOfMCIterations, "\n" )
  predictionRefineStage[i,,,] <- modelRefineStage$predict( dataRefineStage )[1,,,,1]
  }
predictionRefineStage <- apply( predictionRefineStage, c( 2, 3, 4 ), mean )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "    Refine step 4: Average monte carlo results and write probability mask image.\n" )
startTime <- Sys.time()
predictionRefineStageArray <- array( data = 0, dim = dim( imageResampled ) )
predictionRefineStageArray[lower[1]:upper[1],lower[2]:upper[2],lower[3]:upper[3]] <- predictionRefineStage
probabilityMaskRefineStageResampled <- as.antsImage( predictionRefineStageArray ) %>% antsCopyImageInfo2( imageResampled )
probabilityMaskRefineStage <- resampleImageToTarget( probabilityMaskRefineStageResampled, image )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n\n" )

cat( "Renormalize to native space" )
startTime <- Sys.time()
probabilityMask <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
  probabilityMaskRefineStage, image )
antsImageWrite( probabilityMask, outputFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )


endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "  (Total elapsed time:", elapsedTimeTotal, "seconds)\n" )
