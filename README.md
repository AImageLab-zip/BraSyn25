# BraSyn25
Method for cross-modal generation of MRI modalities, submitted to the BraTS 2025 challenge (BraSyn task)

# How to run this
Remember to:
- Have an internet connection (checkpoint downloading would fail otherwise)
- Select a meaningful configuration of RUN_ID and VIEW in the Dockerfile
- Mount the right input/output directories
- Test the Docker before submission!

# Docker tutorial:
- If your machine runs Windows or ...macOS (??????????????????') you should eradicate your installation and install Ubuntu / use a VM (cringe option)
- Check if enroot is installed. If not, follow this tutorial: https://github.com/NVIDIA/enroot/blob/master/doc/installation.md
- After cloning the repo and checking everything is ok, cd to the project directory and build locally the docker
local:   $ sudo docker build -t brats_2025 .
- Uh oh! There is a 69% probability of you forgetting to install docker
- Follow this https://docs.docker.com/desktop/setup/install/linux/ubuntu/
- Convert the docker into a sqsh file, compatible with enroot
local:   $ sudo enroot import -o brats_2025.sqsh dockerd://brats_2025:latest
- Transfer brats_2025.sqsh to the cluster using sftp/something fancier
- Now get a job running on a node
cluster: $ srun --gres=gpu:1 --time=4:00:00 --partition=all_serial --cpus-per-task=8 --mem=20G --account=tesi_ocarpentiero --nodelist=ailb-login-03 --pty bash 
- Time to run the thing!
cluster: $ enroot remove brats_2025 # IF IT'S NOT YOUR FIRST TIME FOLLOWING THIS TUTORIAL <3
cluster: $ enroot create --name brats_2025 path/to/your/beautiful/file.sqsh
- enroot start \
  --mount /work/tesi_ocarpentiero/brats3d/pseudo_random/original:/input:ro,x-create=dir \
  --mount /work/tesi_ocarpentiero/brats3d/pseudo_random/recon:/output:rw,x-create=dir \
  brats-2025
- Watch as this tutorial fails you completely since everything has to go wrong





