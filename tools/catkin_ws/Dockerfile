FROM ubuntu:melodic
ADD file:40290d9a94ae76c35ab1f57178130ce1c5b976e34a91e77472ecf7e945ab64f9 in / 
CMD ["bash"]
/bin/sh -c echo 'Etc/UTC' > /etc/timezone &&     ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime &&     apt-get update &&     apt-get install -q -y --no-install-recommends tzdata &&     rm -rf /var/lib/apt/lists/*
/bin/sh -c apt-get update && apt-get install -q -y --no-install-recommends     dirmngr     gnupg2     && rm -rf /var/lib/apt/lists/*
/bin/sh -c echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list
/bin/sh -c apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO=melodic
/bin/sh -c apt-get update &&
COPY file:cbbaa0f5d6a276512315f5b4d7347e94a120cefbda9058ebb0d678847ff4837f in / 
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
/bin/sh -c apt-get update && apt-get install --no-install-recommends -y     build-essential     python-rosdep     python-rosinstall     python-vcstools     && rm -rf /var/lib/apt/lists/*
/bin/sh -c rosdep init &&   rosdep update --rosdistro $ROS_DISTRO
/bin/sh -c apt-get update && apt-get install -y --no-install-recommends     ros-melodic-ros-base=1.4.1-0*     && rm -rf /var/lib/apt/lists/*
/bin/sh -c apt-get update && apt-get install -y --no-install-recommends     ros-melodic-robot=1.4.1-0*     && rm -rf /var/lib/apt/lists/*
RUN /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     ros-melodic-desktop=1.4.1-0*     && rm -rf /var/lib/apt/lists/* # buildkit
RUN /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     ros-melodic-desktop-full=1.4.1-0*     && rm -rf /var/lib/apt/lists/* # buildkit
