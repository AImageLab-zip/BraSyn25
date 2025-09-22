# Do everything on your pc
````
sudo docker login --username your_username --password your_secret_auth_token ghcr.io
sudo docker build -t brats_2025 .
sudo docker tag brats_2025 ghcr.io/github_username/brats_2025
sudo docker push ghcr.io/github_username/brats_2025
````

Now go to its github page and get the digest.\
Look on the BraTS tutorial for clearer instructions.

Add the ghcr.io/github_username/brats_2025 link to the hf-gan-zip project.\
It doesn't work still, modify the committer.py file and run it.\
Refresh the hf-gan-zip page, it should have a commit now.\
Submit the docker to the task 8 challenge queue\.

