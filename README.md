# inception_v3

cd ~/catkin_ws/src/tfClassifier/image_classification

python retrain.py --model_dir ./inception --image_dir ~/catkin_ws/src/inception_v3/dataset ./output_dir --output_graph ~/catkin_ws/src/inception_v3/model --output_labels ~/catkin_ws/src/inception_v3/model --how_many_training_steps 1000

