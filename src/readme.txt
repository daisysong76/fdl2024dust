
conda activate dust
gsutil ls


gcloud init
gcloud auth login
gcloud config set project 71399020596 # PROJECT_ID 

https://console.cloud.google.com/compute/instances

gcloud compute ssh INSTANCE_NAME --zone=ZONE

gcloud version
gcloud components update
gcloud config list


Run the training for the video model:
python step_fusion_3d.py

Run the training for the time-series model:
python step_fusion_1d.py

Extract video features:
python step_fusion_3d_get_feature.py

Update the time-series model with LoRA:
python step_fusion.py

Additional Notes
Implement LoRA: Ensure you have a proper implementation of LoRA. If you don't have one, you can refer to the LoRA paper and create your own based on the described method.
Hyperparameter Tuning: Fine-tuning with LoRA may require different hyperparameters compared to the original model. Experiment with different values to achieve the best performance.
Logging and Monitoring: Use tools like TensorBoard to monitor the training process and make necessary adjustments.