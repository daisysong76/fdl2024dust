Run the training for the video model:

bash
Copy code
python step_fusion_3d.py
Run the training for the time-series model:

bash
Copy code
python step_fusion_1d.py
Extract video features:

bash
Copy code
python step_fusion_3d_get_feature.py
Update the time-series model with LoRA:

bash
Copy code
python step_fusion.py
Additional Notes
Implement LoRA: Ensure you have a proper implementation of LoRA. If you don't have one, you can refer to the LoRA paper and create your own based on the described method.
Hyperparameter Tuning: Fine-tuning with LoRA may require different hyperparameters compared to the original model. Experiment with different values to achieve the best performance.
Logging and Monitoring: Use tools like TensorBoard to monitor the training process and make necessary adjustments.