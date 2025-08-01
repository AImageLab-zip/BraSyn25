# BraSyn25
Method for cross-modal generation of MRI modalities, submitted to the BraTS 2025 challenge (BraSyn task)

# How to run this
Remember to:
- Have an internet connection (checkpoint downloading would fail otherwise)
- Select a meaningful configuration of RUN_ID and VIEW in the Dockerfile
- Mount the right input/output directories
- Test the Docker before submission!

# How to run on cluster 
srun docker run \
  -v /work/tesi_ocarpentiero/brats3d/pseudo_random/original:/input:ro
  -v /work/tesi_ocarpentiero/brats3d/pseudo_random/recon:/output
  image-name \

