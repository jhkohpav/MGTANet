# MGTANet
Pytorch implemetation of the paper
* **Title**: MGTANet: Encoding LiDAR Point Cloud Sequence Using Long Short-Term Motion-Guided Temporal Attention for 3D Object Detection
* **Author**: Junho Koh*, Junhyung Lee*, Youngwoo Lee, Jaekyum Kim, Jun Won Choi (* indicates equal contribution)
## Abstract
Most scanning LiDAR sensors generate a sequence of point clouds in real-time. While conventional 3D object detectors use a set of unordered LiDAR points acquired over a fixed time interval, recent studies have revealed that substantial performance improvement can be achieved by exploiting the *spatio-temporal context* existing in a sequence of LiDAR point sets. In this paper, we propose a novel architecture for 3D object detection, which can encode LiDAR point cloud sequences acquired by multiple successive scans. The encoding process of the point cloud sequence is performed on two different time scales. We first design a *short-term motion-aware voxel encoding* that captures the temporal changes of point clouds driven by the motion of objects in each voxel. We also propose *long-term motion-guided bird's eye view (BEV) feature enhancement* that adaptively aligns and aggregates the BEV features obtained by the short-term voxel encoding. The long-term BEV feature encoding facilitates a motion-guided attention mechanism that determines the position offsets and the weights of the features to temporally aggregate them based on the dynamic motion context inferred from the sequence of BEV feature maps. The experiments conducted on the public nuScenes benchmark demonstrate that the proposed 3D object detector combining the aforementioned two components achieves a significant improvement in performance compared to a baseline method based on a single point set. Moreover, it also demonstrates state-of-the-art performance for certain 3D object detection categories.

We will open the code soon.
