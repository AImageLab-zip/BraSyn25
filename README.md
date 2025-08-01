# BraSyn25
Method for cross-modal generation of MRI modalities, submitted to the BraTS 2025 challenge (BraSyn task)

# How to run this
Remember to:
- Have an internet connection (checkpoint downloading would fail otherwise)
- Select a meaningful configuration of RUN_ID and VIEW in the Dockerfile
- Mount the right input/output directories
- Test the Docker before submission!

# Docker tutorial:
local:   $ sudo docker build -t brats_2025 .
local:   $ sudo enroot import -o brats_2025.sqsh dockerd://brats_2025:latest
- Transfer brats_2025.sqsh to the cluster using sftp
cluster: $ srun --container-image=enroot:brats_2025.sqsh




