# Deep Learning for Localization and Mapping

![image](image/concept_figure.png)
This repository is a collection of deep learning based localization and mapping approaches. 

## News
### Update: Jun-22-2020
- We released our survey paper "A Survey on Deep Learning for localization and mapping: Towards the Age of Spatial Machine Intelligence".

## TO DO
- Global Localization
- SLAM

## Category
- [Odometry Estimation](#Odometry-Estimation)
  - [Visual Odometry](#Visual-Odometry)
  - [Visual-Inertial Odometry](#Visual-Inertial-Odometry)
  - [Inertial Odometry](#Inertial-Odometry)
  - [LIDAR Odometry](#LIDAR-Odometry)
- [Mapping](#Mapping)
  - [Geometric Mapping](#Geometric-Mapping)
  - [Semantic Mapping](#Semantic-Mapping)
  - [General Mapping](#General-Mapping)
- [Global localization](#Global-Localization)
  - [2D-to-2D Localization](#2D-to-2D-Localization)
  - [2D-to-3D Localization](#2D-to-3D-Localization)
  - [3D-to-3D Localization](#3D-to-3D-Localization)
- [Simultaneous Localization and Mapping (SLAM)](#SLAM)
  - [Local Optimization](#Local-Optimization)
  - [Global Optimization](#Global-Optimization)
  - [Keyframe and Loop-closure Detection](#Keyframe-and-Loop-closure-Detection)
  - [Uncertainty Estimation](#Uncertainty-Estimation)
  
## Categorized by Topic
*The Date in the table denotes the publication date (e.g. date of conference).
### Odometry Estimation
#### Visual Odometry
| Models   |Date| Publication| Paper | Code |
|----------|----|------------|------|---|
| Konda et al. | 2015 | VISAPP | [Learning visual odometry with a convolutional network](https://www.iro.umontreal.ca/~memisevr/pubs/VISAPP2015.pdf) | |
| Costante et al. | 2016 | RA-L | [Exploring Representation Learning With CNNs for Frame-to-Frame Ego-Motion Estimation](https://ieeexplore.ieee.org/document/7347378) | |
| Backprop KF | 2016 | NeurIPS | [Backprop KF: Learning Discriminative Deterministic State Estimators](https://arxiv.org/abs/1605.07148) | |
| DeepVO | 2017 | ICRA | [DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1709.08429) | |
| SfmLearner | 2017 | CVPR | [unsupervised learning of depth and ego-motion from video](http://openaccess.thecvf.com/content_cvpr_2018/papers/Mahjourian_Unsupervised_Learning_of_CVPR_2018_paper.pdf) | [TF](https://github.com/tinghuiz/SfMLearner) [PT](https://github.com/ClementPinard/SfmLearner-Pytorch)|
| Yin et al. | 2017 | ICCV | [Scale Recovery for Monocular Visual Odometry Using Depth Estimated With Deep Convolutional Neural Fields](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yin_Scale_Recovery_for_ICCV_2017_paper.pdf) | |
| UnDeepVO | 2018 | ICRA | [UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning](https://arxiv.org/abs/1709.06841) | |
| Barnes et al. | 2018 | ICRA | [Driven to Distraction: Self-Supervised Distractor Learning for Robust Monocular Visual Odometry in Urban Environments](https://arxiv.org/abs/1711.06623) | |
| GeoNet | 2018 | CVPR | [GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose](https://arxiv.org/abs/1803.02276) | [TF](https://github.com/yzcjtr/GeoNet) |
| Zhan et al. | 2018 | CVPR | [Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction](https://arxiv.org/abs/1803.03893) | [Caffe](https://github.com/Huangying-Zhan/Depth-VO-Feat) |
| DPF | 2018 | RSS | [Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors](https://arxiv.org/abs/1805.11122) | [TF](https://github.com/tu-rbo/differentiable-particle-filters) |
| Yang et al. | 2018 | ECCV | [Deep Virtual Stereo Odometry: Leveraging Deep Depth Prediction for Monocular Direct Sparse Odometry](https://arxiv.org/abs/1807.02570) | |
| Zhao et al. | 2018 | IROS | [Learning monocular visual odometry with dense 3d mapping from dense 3d flow](https://arxiv.org/abs/1803.02286) | |
| Turan et al. | 2018 | IROS | [Unsupervised Odometry and Depth Learning for Endoscopic Capsule Robots](https://arxiv.org/pdf/1803.01047.pdf) | |
| Struct2Depth | 2019 | AAAI | [Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos](https://arxiv.org/abs/1811.06152) | [TF](https://github.com/tensorflow/models/tree/master/research/struct2depth) |
| Saputra et al.| 2019 | ICRA | [Learning monocular visual odometry through geometry-aware curriculum learning](https://arxiv.org/abs/1903.10543) | |
| GANVO | 2019 | ICRA | [GANVO: Unsupervised deep monocular visual odometry and depth estimation with generative adversarial networks](https://arxiv.org/abs/1809.05786) | |
| CNN-SVO | 2019 | ICRA | [CNN-SVO: Improving the Mapping in Semi-Direct Visual Odometry Using Single-Image Depth Prediction](https://ieeexplore.ieee.org/document/8794425) | [ROS](https://github.com/yan99033/CNN-SVO) |
| Li et al. | 2019 | ICRA | [Pose graph optimization for unsupervised monocular visual odometry](https://arxiv.org/abs/1903.06315) | |
| Xue et al.| 2019 | CVPR | [Beyond tracking: Selecting memory and refining poses for deep visual odometry](https://arxiv.org/abs/1904.01892) | |
| Wang et al.| 2019 | CVPR | [Recurrent neural network for (un-) supervised learning of monocular video visual odometry and depth](https://arxiv.org/abs/1904.07087) | |
| Li et al. | 2019 | ICCV | [Sequential adversarial learning for self-supervised deep visual odometry](https://arxiv.org/abs/1908.08704) | |
| Saputra et al. | 2019 | ICCV | [Distilling knowledge from a deep pose regressor network](https://arxiv.org/abs/1908.00858) | |
| Gordon et al. | 2019 | ICCV | [Depth from videos in the wild: Unsupervised monocular depth learning from unknown cameras](https://arxiv.org/abs/1904.04998) | [TF](https://github.com/google-research/google-research/tree/master/depth_from_video_in_the_wild) |
|  Koumis et al. | 2019 | IROS | [Estimating Metric Scale Visual Odometry from Videos using 3D Convolutional Networks](https://jpreiss.github.io/pubs/Koumis_Preiss_3DCVO_IROS2019.pdf) | |
| Bian et al. | 2019 | NeurIPS | [Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video](https://papers.nips.cc/paper/8299-unsupervised-scale-consistent-depth-and-ego-motion-learning-from-monocular-video.pdf) | [PT](https://github.com/JiawangBian/SC-SfMLearner-Release) |
| D3VO | 2020 | CVPR | [D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry](https://arxiv.org/pdf/2003.01060.pdf#page=9&zoom=100,412,902) | |
| Jiang et al. | 2020 | CVPR | [Joint Unsupervised Learning of Optical Flow and Egomotion with Bi-Level Optimization](https://arxiv.org/pdf/2002.11826.pdf) | |


#### Visual-Inertial Odometry
| Models   |Date| Publication| Paper | Code |
|----------|----|------------|------|---|
| VINet | 2017 | AAAI | [VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem](https://arxiv.org/abs/1701.08376) | |
| VIOLearner | 2019 | TPAMI | [Unsupervised deep visual-inertial odometry with online error correction for rgb-d imagery](https://ieeexplore.ieee.org/document/8691513) | |
| SelectFusion | 2019 | CVPR | [Selective Sensor Fusion for Neural Visual-Inertial Odometry](https://arxiv.org/abs/1903.01534) | |
| DeepVIO | 2019 | IROS | [DeepVIO: Self-supervised deep learning of monocular visual inertial odometry using 3d geometric constraints](https://arxiv.org/abs/1906.11435) | |


#### Inertial Odometry
| Models   |Date| Publication| Paper | Code |
|----------|----|------------|------|---|
| IONet | 2018 | AAAI | [IONet: Learning to Cure the Curse of Drift in Inertial Odometry](https://arxiv.org/abs/1802.02209) | |
| RIDI | 2018 | ECCV | [RIDI: Robust IMU Double Integration](https://arxiv.org/abs/1712.09004) | [Py](https://github.com/higerra/ridi_imu) |
| Wagstaff et al. | 2018 | IPIN | [LSTM-Based Zero-Velocity Detection for Robust Inertial Navigation](https://ieeexplore.ieee.org/abstract/document/8533770) | [PT](https://github.com/utiasSTARS/pyshoe) |
| Cortes et al. | 2019 | MLSP | [Deep Learning Based Speed Estimation for Constraining Strapdown Inertial Navigation on Smartphones](https://ieeexplore.ieee.org/abstract/document/8516710) | |
| MotionTransformer| 2019 | AAAI | [MotionTransformer: Transferring Neural Inertial Tracking between Domains](https://www.aaai.org/ojs/index.php/AAAI/article/view/4802) | |
| AbolDeepIO | 2019 | TITS | [AbolDeepIO: A Novel Deep Inertial Odometry Network for Autonomous Vehicles](https://ieeexplore.ieee.org/abstract/document/8693766) | |
| Brossard et al. | 2019 | ICRA | [Learning wheel odometry and imu errors for localization](https://hal.archives-ouvertes.fr/hal-01874593/document) | |
| OriNet | 2019 | RA-L | [OriNet: Robust 3-D Orientation Estimation With a Single Particular IMU](https://ieeexplore.ieee.org/abstract/document/8931590) | [PT](https://github.com/mbrossar/denoise-imu-gyro) |
| L-IONet | 2020| IoT-J | [Deep Learning based Pedestrian Inertial Navigation: Methods, Dataset and On-Device Inference](https://arxiv.org/abs/2001.04061) | |

#### LIDAR Odometry
| Models   |Date| Publication| Paper | Code |
|----------|----|------------|------|---|
| Velas et al. | 2018 | ICARSC | [CNN for IMU Assisted Odometry Estimation using Velodyne LiDAR](https://arxiv.org/abs/1712.06352) | | 
| LO-Net | 2019 | CVPR | [LO-Net: Deep Real-time Lidar Odometry](https://arxiv.org/abs/1904.08242) | |
| DeepPCO | 2019 | IROS | [DeepPCO: End-to-End Point Cloud Odometry through Deep Parallel Neural Network](https://arxiv.org/abs/1910.11088) | |
| Valente et al. | 2019 | IROS | [Deep sensor fusion for real-time odometry estimation](https://ieeexplore.ieee.org/document/8967803) | |

### Mapping
#### Geometric Mapping
##### Depth Representation
| Models   |Date| Publication| Paper | Code |
|----------|----|------------|------|---|
| Eigen et al. | 2014 | NeurIPS | [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf) | |

##### Voxel Representation
| Models   |Date| Publication| Paper | Code |
|----------|----|------------|------|---|
| SurfaceNet | 2017 | CVPR | [SurfaceNet: An End-to-end 3D Neural Network for Multiview Stereopsis](https://arxiv.org/abs/1708.01749) | |
| Dai et al. | 2017 | CVPR | [Shape completion using 3d-encoder-predictor cnns and shape synthesis](https://arxiv.org/abs/1612.00101) | |
| Hane et al. | 2017 | 3DV | [Hierarchical surface prediction for 3d object reconstruction](https://arxiv.org/abs/1704.00710) | |
| OctNetFusion | 2017 | 3DV | [Octnetfusion: Learning depth fusion from data](https://arxiv.org/abs/1704.01047) | |
| OGN | 2017 | ICCV | [Octree generating networks: Efficient convolutional architectures for high-resolution 3d outputs](https://arxiv.org/abs/1703.09438) | |
| Kar et al. | 2017 | NeurIPS | [Learning a multi-view stereo machine](https://arxiv.org/abs/1708.05375) | |
| RayNet | 2018 | CVPR | [RayNet: Learning Volumetric 3D Reconstruction with Ray Potentials](https://arxiv.org/abs/1901.01535) | |


##### Point Representation
| Models   |Date| Publication| Paper | Code |
|----------|----|------------|------|---|
| Fan et al. | 2017 | CVPR | [A point set generation network for 3d object reconstruction from a single image](https://arxiv.org/abs/1612.00603) | |

##### Mesh Representation
| Models   |Date| Publication| Paper | Code |
|----------|----|------------|------|---|
| Ladicky et al. | 2017 | ICCV | [From point clouds to mesh using regression](https://ieeexplore.ieee.org/document/8237682) | |
| Mukasa et al. | 2017 | ICCVW | [3d scene mesh from cnn depth predictions and sparse monocular slam](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Mukasa_3D_Scene_Mesh_ICCV_2017_paper.pdf) | |
| Wang et al. | 2018 | ECCV | [Pixel2mesh: Generating 3d mesh models from single rgb images](https://arxiv.org/abs/1804.01654) | |
| Groueix et al. | 2018 | CVPR | [AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation](https://arxiv.org/abs/1802.05384) | |
| Scan2Mesh | 2019 | CVPR | [Scan2mesh: From unstructured range scans to 3d meshes](https://arxiv.org/abs/1811.10464) | |
| Bloesch et al. | 2019 | ICCV | [Learning meshes for dense visual SLAM](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/dyson-robotics-lab/mbloesch_etal_iccv2019.pdf) | |

#### Semantic Mapping

#### General Mapping

### Global Localization
#### 2D-to-2D Localization

#### 2D-to-3D Localization

#### 3D-to-3D Localization

### SLAM

#### Local Optimization

#### Global Optimization

#### Keyframe and Loop-closure Detection

#### Uncertainty Estimation

This list is maintained by [Changhao Chen](http://www.cs.ox.ac.uk/people/changhao.chen/website/) and [Bing Wang](http://www.cs.ox.ac.uk/people/bing.wang/), Department of Computer Science, University of Oxford.

Please contact them (email: changhao.chen@cs.ox.ac.uk; bing.wang@cs.ox.ac.uk), if you have any question or would like to add your work on this list.
