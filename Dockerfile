FROM saikishorpal/kinetic-tensorflow:latest

# Setting Catkin Workspace
RUN /bin/bash -c "apt-get update &&\
    source /opt/ros/kinetic/setup.bash &&\
    mkdir -p ~/catkin_ws/src &&\
    cd ~/catkin_ws/src &&\
    catkin_init_workspace &&\
    cd ~/catkin_ws &&\
    catkin_make &&\
    source devel/setup.bash"

# Install git
RUN /bin/bash -c "apt-get install -y git"

# Key Update
RUN /bin/bash -c "apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116 &&\
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 &&\
    apt-get clean && apt-get update"

# Install ROS camera driver & dependencies install
RUN /bin/bash -c "apt-get install -y ros-kinetic-cv-bridge ros-kinetic-opencv3"

# Install cv camera
RUN /bin/bash -c "apt-get -y install ros-kinetic-cv-camera"

# Install ros_inception_v3
RUN /bin/bash -c "apt-get update &&\
    source /opt/ros/kinetic/setup.bash &&\
    cd ~/catkin_ws/src &&\
    git clone https://github.com/kungfrank/ros_inception_v3.git &&\
    cd ~/catkin_ws &&\
    catkin_make &&\
    source devel/setup.bash"

# Install tf classifier
RUN /bin/bash -c "apt-get update &&\
    source /opt/ros/kinetic/setup.bash &&\
    cd ~/catkin_ws/src &&\
    git clone https://github.com/akshaypai/tfClassifier &&\
    cd ~/catkin_ws &&\
    catkin_make &&\
    source devel/setup.bash"

# Install flower_photos dataset
RUN /bin/bash -c "cd ~ &&\
    curl -O http://download.tensorflow.org/example_images/flower_photos.tgz &&\
    tar xzf flower_photos.tgz"

# Install Python Package for image process
RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash &&\
    pip install imutils"
