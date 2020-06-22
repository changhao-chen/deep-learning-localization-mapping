# Deep Learning for Localization and Mapping

![image](image/concept_figure.png)
This repository is a collection of deep learning based localization and mapping approaches. 

## News
## Update: Jun-22-2020
- We released our survey paper "A Survey on Deep Learning for localization and mapping: Towards the Age of Spatial Machine Intelligence".

## Category
-[Odometry Estimation](#odometry)
-[Mapping](#mapping)
-[Global localization](#global-localization)
-[Simultaneous Localization and Mapping (SLAM)](#slam)

## Categorized by Topic
### global0localization
*The Year/Month in the table denotes the publication date (date of conference).
| Models   |Year| Publication| Paper | Code |
|----------|----|------------|------|---|
| PoseNet  | 2015/12 | ICCV | [PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](https://arxiv.org/abs/1505.07427)  | |
| Bayesian PoseNet  | 2016/05 | ICRA | [Modelling uncertainty in deep learning for camera relocalization](https://arxiv.org/abs/1509.05909)  | |
| VidLoc   | 2017/07 | CVPR | [VidLoc: A Deep Spatio-Temporal Model for 6-DoF Video-Clip Relocalization](https://arxiv.org/abs/1702.06521)  | |
| PoseNet+ | 2017/07 | CVPR | [Geometric loss functions for camera pose regression with deep learning](http://openaccess.thecvf.com/content_cvpr_2017/html/Kendall_Geometric_Loss_Functions_CVPR_2017_paper.html) | |
| DSAC | 2017/07 | CVPR | [DSAC - Differentiable RANSAC for Camera Localization](http://openaccess.thecvf.com/content_cvpr_2017/html/Brachmann_DSAC_-_Differentiable_CVPR_2017_paper.html) | | 
| Kim et al. | 2017/07 | CVPR | [Learned Contextual Feature Reweighting for Image Geo-Localization](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kim_Learned_Contextual_Feature_CVPR_2017_paper.pdf) | | 
| Naseer et al. | 2017/09 | IROS | [Deep Regression for Monocular Camera-based 6-DoF Global Localization in Outdoor Environments](http://ais.informatik.uni-freiburg.de/publications/papers/naseer17iros.pdf)| |
| LSTM-PoseNet | 2017/10| ICCV | [Image-based localization using lstms for structured feature correlation](https://arxiv.org/abs/1611.07890) | |
| Laskar et al. | 2017/10| ICCV Workshops | [Camera Relocalization by Computing Pairwise Relative Poses Using Convolutional Neural Network](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Laskar_Camera_Relocalization_by_ICCV_2017_paper.pdf) | |
| X-View | 2018/02 | RA-L | [X-View: Graph-Based Semantic Multi-View Localization](https://arxiv.org/abs/1709.09905)| |
| Hourglass PoseNet | 2017/10| ICCV Workshops | [Image-based localization using hourglass networks](http://openaccess.thecvf.com/content_ICCV_2017_workshops/w17/html/Melekhov_Image-Based_Localization_Using_ICCV_2017_paper.html) | |
| VLocNet | 2018/05 | ICRA | [Deep auxiliary learning for visual localization and odometry](https://arxiv.org/abs/1803.03642)| |
| ContextualNet | 2018/05 | ICRA | [ContextualNet: Exploiting Contextual Information using LSTMs to Improve Image-based Localization](https://ieeexplore.ieee.org/document/8461124)| |
| MapNet | 2018/06 | CVPR | [Geometry-Aware Learning of Maps for Camera Localization](https://arxiv.org/abs/1712.03342)| |
| DSAC++ | 2018/06 | CVPR | [Learning less is more-6d camera localization via 3d surface regression](https://arxiv.org/abs/1711.10228) | |
| Sattler et al. | 2018/06 | CVPR | [Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions](http://openaccess.thecvf.com/content_cvpr_2018/html/Sattler_Benchmarking_6DOF_Outdoor_CVPR_2018_paper.html) | |
| Schonberger et al.| 2018/06 | CVPR | [Semantic Visual Localization](http://openaccess.thecvf.com/content_cvpr_2018/papers/Schonberger_Semantic_Visual_Localization_CVPR_2018_paper.pdf) | |
| InLoc | 2018/06 | CVPR | [InLoc: Indoor Visual Localization with Dense Matching and View Synthesis](http://openaccess.thecvf.com/content_cvpr_2018/papers/Taira_InLoc_Indoor_Visual_CVPR_2018_paper.pdf) | |
| Li et al. | 2018/07 | RSS | [Full-Frame Scene Coordinate Regression for Image-Based Localization](https://arxiv.org/abs/1802.03237) | |
| DSAC++ angle loss | 2018/09 | ECCV | [Scene coordinate regression with angle-based reprojection loss for camera relocalization](http://openaccess.thecvf.com/content_eccv_2018_workshops/w16/html/Li_Scene_Coordinate_Regression_with_Angle-Based_Reprojection_Loss_for_Camera_Relocalization_ECCVW_2018_paper.html) | |
| RelocNet | 2018/09 | ECCV | [RelocNet: Continuous Metric Learning Relocalisation using Neural Nets](http://openaccess.thecvf.com/content_ECCV_2018/papers/Vassileios_Balntas_RelocNet_Continous_Metric_ECCV_2018_paper.pdf) | |
| Bui et al. | 2018/09 | BMVC | [Scene Coordinate and Correspondence Learning for Image-Based Localization](https://arxiv.org/abs/1805.08443) | |
| VLocNet++ | 2018/09 | RA-L | [Vlocnet++: Deep multitask learning for semantic visual localization and odometry](https://arxiv.org/abs/1804.08366)| |
| Piasco et al. | 2019/05 | ICRA | [Learning scene geometry for visual localization in challenging conditions](https://hal.archives-ouvertes.fr/hal-02057378/document)| |
| Amini et al. | 2019/05 | ICRA | [Variational End-to-End Navigation and Localization](https://arxiv.org/abs/1811.10119) | |
| Sattler et al. | 2019/06 | CVPR | [Understanding the Limitations of CNN-based Absolute Camera Pose Regression](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sattler_Understanding_the_Limitations_of_CNN-Based_Absolute_Camera_Pose_Regression_CVPR_2019_paper.pdf)| |
| Sarlin et al. | 2019/06 | CVPR | [From Coarse to Fine: Robust Hierarchical Localization at Large Scale](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.pdf)| |
| NG-RANSAC | 2019/06 | CVPR | [Neural-Guided RANSAC: Learning Where to Sample Model Hypotheses](http://openaccess.thecvf.com/content_ICCV_2019/papers/Brachmann_Neural-Guided_RANSAC_Learning_Where_to_Sample_Model_Hypotheses_ICCV_2019_paper.pdf)| |
| Liu et al. | 2019/06 | CVPR | [Lending Orientation to Neural Networks for Cross-view Geo-localization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Lending_Orientation_to_Neural_Networks_for_Cross-View_Geo-Localization_CVPR_2019_paper.pdf)| |
| Weinzaepfel et al. | 2019/06 | CVPR | [Visual Localization by Learning Objects-of-Interest Dense Match Regression](http://openaccess.thecvf.com/content_CVPR_2019/papers/Weinzaepfel_Visual_Localization_by_Learning_Objects-Of-Interest_Dense_Match_Regression_CVPR_2019_paper.pdf)| |
| NG-RANSAC | 2019/06 | CVPR | [Neural-Guided RANSAC: Learning Where to Sample Model Hypotheses](http://openaccess.thecvf.com/content_ICCV_2019/papers/Brachmann_Neural-Guided_RANSAC_Learning_Where_to_Sample_Model_Hypotheses_ICCV_2019_paper.pdf)| |
| SANet | 2019/10 | ICCV | [SANet: scene agnostic network for camera localization](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_SANet_Scene_Agnostic_Network_for_Camera_Localization_ICCV_2019_paper.pdf)| |
| Huang et al. | 2019/10 | ICCV | [Prior guided dropout for robust visual localization in dynamic environments](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Prior_Guided_Dropout_for_Robust_Visual_Localization_in_Dynamic_Environments_ICCV_2019_paper.pdf) | |
| Xue et al. | 2019/10 | ICCV |  [Local supports global: Deep camera relocalization with sequence enhancement](https://arxiv.org/abs/1908.04391)| |
| CamNet | 2019/10 |  ICCV | [Camnet: Coarse-to-fine retrieval for camera re-localization](http://openaccess.thecvf.com/content_ICCV_2019/html/Ding_CamNet_Coarse-to-Fine_Retrieval_for_Camera_Re-Localization_ICCV_2019_paper.html)| |
| ESAC | 2019/10 |  ICCV | [Expert Sample Consensus Applied to Camera Re-Localization](http://openaccess.thecvf.com/content_ICCV_2019/html/Ding_CamNet_Coarse-to-Fine_Retrieval_for_Camera_Re-Localization_ICCV_2019_paper.html)| |
| Taira et al. | 2019/10 |  ICCV | [Is This The Right Place? Geometric-Semantic Pose Verification for Indoor Visual Localization](http://openaccess.thecvf.com/content_ICCV_2019/papers/Taira_Is_This_the_Right_Place_Geometric-Semantic_Pose_Verification_for_Indoor_ICCV_2019_paper.pdf)| |
| Liu et al. | 2019/10 |  ICCV | [Stochastic Attraction-Repulsion Embedding for Large Scale Image Localization](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Stochastic_Attraction-Repulsion_Embedding_for_Large_Scale_Image_Localization_ICCV_2019_paper.pdf)| |
| FGSN | 2019/10 |  ICCV | [Fine-Grained Segmentation Networks: Self-Supervised Segmentation for Improved Long-Term Visual Localization](http://openaccess.thecvf.com/content_ICCV_2019/papers/Larsson_Fine-Grained_Segmentation_Networks_Self-Supervised_Segmentation_for_Improved_Long-Term_Visual_Localization_ICCV_2019_paper.pdf)| |
| Cheng et al. | 2019/10 |  ICCV | [Cascaded Parallel Filtering for Memory-Efficient Image-Based Localization](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cheng_Cascaded_Parallel_Filtering_for_Memory-Efficient_Image-Based_Localization_ICCV_2019_paper.pdf)| |
| Song et al. | 2019/11 | IROS | [Learning Local Feature Descriptor with Motion Attribute for Vision-based Localization](https://arxiv.org/abs/1908.01180) ||
| GN-Net | 2020/01 | RA-L | [GN-Net: The Gauss-Newton Loss for Multi-Weather Relocalization](https://ieeexplore.ieee.org/abstract/document/8954808) | |
| AtLoc | 2020/02 | AAAI | [AtLoc: Attention Guided Camera Localization](https://arxiv.org/abs/1909.03557) | |
| Zhou et al. | 2020/05 | ICRA | [To Learn or Not to Learn: Visual Localization from Essential Matrices](https://arxiv.org/abs/1908.01293) | |

## Visual Odometry
| Models   |Year/Month| Publication| Paper | Code |
|----------|----|------------|------|---|
| Konda et al. | 2015/03 | VISAPP | [Learning visual odometry with a convolutional network](https://www.iro.umontreal.ca/~memisevr/pubs/VISAPP2015.pdf) | |
| Costante et al. | 2016/01 | RA-L | [Exploring Representation Learning With CNNs for Frame-to-Frame Ego-Motion Estimation](https://ieeexplore.ieee.org/document/7347378) | |
| Backprop KF | 2016/12 | NeurIPS | [Backprop KF: Learning Discriminative Deterministic State Estimators](https://arxiv.org/abs/1605.07148) | |
| DeepVO | 2017/05 | ICRA | [DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1709.08429) | |
| SfmLearner | 2017/07 | CVPR | [unsupervised learning of depth and ego-motion from video](http://openaccess.thecvf.com/content_cvpr_2018/papers/Mahjourian_Unsupervised_Learning_of_CVPR_2018_paper.pdf) | |
| Yin et al. | 2017/10 | ICCV | [Scale Recovery for Monocular Visual Odometry Using Depth Estimated With Deep Convolutional Neural Fields](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yin_Scale_Recovery_for_ICCV_2017_paper.pdf) | |
| UnDeepVO | 2018/05 | ICRA | [UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning](https://arxiv.org/abs/1709.06841) | |
| Barnes et al. | 2018/05 | ICRA | [Driven to Distraction: Self-Supervised Distractor Learning for Robust Monocular Visual Odometry in Urban Environments](https://arxiv.org/abs/1711.06623) | |
| GeoNet | 2018/06 | CVPR | [GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose](https://arxiv.org/abs/1803.02276) | |
| Zhan et al. | 2018/06 | CVPR | [Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction](https://arxiv.org/abs/1803.03893) | |
| DPF | 2018/06 | RSS | [Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors](https://arxiv.org/abs/1805.11122) | |
| Yang et al. | 2018/09 | ECCV | [Deep Virtual Stereo Odometry: Leveraging Deep Depth Prediction for Monocular Direct Sparse Odometry](https://arxiv.org/abs/1807.02570) | |
| Zhao et al. | 2018/10 | IROS | [Learning monocular visual odometry with dense 3d mapping from dense 3d flow](https://arxiv.org/abs/1803.02286) | |
| Turan et al. | 2018/10 | IROS | [Unsupervised Odometry and Depth Learning for Endoscopic Capsule Robots](https://arxiv.org/pdf/1803.01047.pdf) | |
| Saputra et al.| 2019/05 | ICRA | [Learning monocular visual odometry through geometry-aware curriculum learning](https://arxiv.org/abs/1903.10543) | |
| GANVO | 2019/05 | ICRA | [GANVO: Unsupervised deep monocular visual odometry and depth estimation with generative adversarial networks](https://arxiv.org/abs/1809.05786) | |
| Li et al. | 2019/05 | ICRA | [Pose graph optimization for unsupervised monocular visual odometry](https://arxiv.org/abs/1903.06315) | |
| Xue et al.| 2019/06 | CVPR | [Beyond tracking: Selecting memory and refining poses for deep visual odometry](https://arxiv.org/abs/1904.01892) | |
| Wang et al.| 2019/06 | CVPR | [Recurrent neural network for (un-) supervised learning of monocular video visual odometry and depth](https://arxiv.org/abs/1904.07087) | |
| Li et al. | 2019/10 | ICCV | [Sequential adversarial learning for self-supervised deep visual odometry](https://arxiv.org/abs/1908.08704) | |
| Saputra et al. | 2019/10 | ICCV | [Distilling knowledge from a deep pose regressor network](https://arxiv.org/abs/1908.00858) | |
|  Koumis et al. | 2019/11 | IROS | [Estimating Metric Scale Visual Odometry from Videos using 3D Convolutional Networks](https://jpreiss.github.io/pubs/Koumis_Preiss_3DCVO_IROS2019.pdf) | |
| Bian et al. | 2019/12 | NeurIPS | [Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video](https://papers.nips.cc/paper/8299-unsupervised-scale-consistent-depth-and-ego-motion-learning-from-monocular-video.pdf) | |
| D3VO | 2020/06 | CVPR | [D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry](https://arxiv.org/pdf/2003.01060.pdf#page=9&zoom=100,412,902) | |
| Jiang et al. | 2020/06 | CVPR | [Joint Unsupervised Learning of Optical Flow and Egomotion with Bi-Level Optimization](https://arxiv.org/pdf/2002.11826.pdf) | |


## Visual-Inertial Odometry
| Models   |Year/Month| Publication| Paper | Code |
|----------|----|------------|------|---|
| VINet | 2017/02 | AAAI | [VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem](https://arxiv.org/abs/1701.08376) | |
| VIOLearner | 2019/04 | TPAMI | [Unsupervised deep visual-inertial odometry with online error correction for rgb-d imagery](https://ieeexplore.ieee.org/document/8691513) | |
| SelectFusion | 2019/05 | CVPR | [Selective Sensor Fusion for Neural Visual-Inertial Odometry](https://arxiv.org/abs/1903.01534) | |
| DeepVIO | 2019/11 | IROS | [DeepVIO: Self-supervised deep learning of monocular visual inertial odometry using 3d geometric constraints](https://arxiv.org/abs/1906.11435) | |


## Inertial Positioning (Odometry)
| Models   |Year/Month| Publication| Paper | Code |
|----------|----|------------|------|---|
| IONet | 2018/02 | AAAI | [IONet: Learning to Cure the Curse of Drift in Inertial Odometry](https://arxiv.org/abs/1802.02209) | |
| RIDI | 2018/09 | ECCV | [RIDI: Robust IMU Double Integration](https://arxiv.org/abs/1712.09004) | |
| Wagstaff et al. | 2018/09 | IPIN | [LSTM-Based Zero-Velocity Detection for Robust Inertial Navigation](https://ieeexplore.ieee.org/abstract/document/8533770) | |
| Cortes et al. | 2019/09 | MLSP | [Deep Learning Based Speed Estimation for Constraining Strapdown Inertial Navigation on Smartphones](https://ieeexplore.ieee.org/abstract/document/8516710) | |
| MotionTransformer| 2019/01 | AAAI | [MotionTransformer: Transferring Neural Inertial Tracking between Domains](https://www.aaai.org/ojs/index.php/AAAI/article/view/4802) | |
| AbolDeepIO | 2019/04 | TITS | [AbolDeepIO: A Novel Deep Inertial Odometry Network for Autonomous Vehicles](https://ieeexplore.ieee.org/abstract/document/8693766) | |
| Brossard et al. | 2019/05 | ICRA | [Learning wheel odometry and imu errors for localization](https://hal.archives-ouvertes.fr/hal-01874593/document) | |
| OriNet | 2019/12 | RA-L | [OriNet: Robust 3-D Orientation Estimation With a Single Particular IMU](https://ieeexplore.ieee.org/abstract/document/8931590) | |
| L-IONet | 2020/01| IOT-J | [Deep Learning based Pedestrian Inertial Navigation: Methods, Dataset and On-Device Inference](https://arxiv.org/abs/2001.04061) | |

## LIDAR Positioning (Odometry)
| Models   |Year/Month| Publication| Paper | Code |
|----------|----|------------|------|---|
| DeLS-3D | 2018/06 | CVPR | [DeLS-3D: Deep Localization and Segmentation with a 3D Semantic Map](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_DeLS-3D_Deep_Localization_CVPR_2018_paper.pdf) | |
| Barsan et al. | 2018/09 | CoRL | [Learning to Localize Using a LiDAR Intensity Map](https://pdfs.semanticscholar.org/dde4/b101c74d8922be2b94db9490ba1bfc8d88ea.pdf) | |
| LO-Net | 2019/06 | CVPR | [LO-Net: Deep Real-time Lidar Odometry](https://arxiv.org/abs/1904.08242) | |
| L3-Net | 2019/06 | CVPR | [L3-Net: Towards Learning based LiDAR Localization for Autonomous Driving](https://songshiyu01.github.io/pdf/L3Net_W.Lu_Y.Zhou_S.Song_CVPR2019.pdf) | |
| DeepPCO | 2019/11 | IROS | [DeepPCO: End-to-End Point Cloud Odometry through Deep Parallel Neural Network](https://arxiv.org/abs/1910.11088) | |
| Valente et al. | 2019/11 | IROS | [Deep sensor fusion for real-time odometry estimation](https://ieeexplore.ieee.org/document/8967803) | |

## Map Representation
| Models   |Year/Month| Publication| Paper | Code |
|----------|----|------------|------|---|
| CodeSLAM | 2018/05 | CVPR | [CodeSLAM - Learning a Compact, Optimisable Representation for Dense Visual SLAM](https://arxiv.org/abs/1804.00874) | |
| GEN-SLAM | 2019/05 | ICRA | [GEN-SLAM: Generative Modeling for Monocular Simultaneous Localization and Mapping](https://arxiv.org/abs/1902.02086) | |

## SLAM Backend
| Models   |Year/Month| Publication| Paper | Code |
|----------|----|------------|------|---|
| LS-Net | 2018/09 | ECCV | [LS-Net: Learning to Solve Nonlinear Least Squares for Monocular Stereo](https://arxiv.org/abs/1809.02966) | |
| BA-Net | 2019/04| ICLR | [BA-Net: Dense bundle adjustment network](https://arxiv.org/abs/1806.04807) | |
| Sheng et al. | 2019/10 | ICCV | [Unsupervised collaborative learning of keyframe detection and visual odometry towards monocular deep slam](http://openaccess.thecvf.com/content_ICCV_2019/papers/Sheng_Unsupervised_Collaborative_Learning_of_Keyframe_Detection_and_Visual_Odometry_Towards_ICCV_2019_paper.pdf) | |

This list is maintained by [Changhao Chen](http://www.cs.ox.ac.uk/people/changhao.chen/website/) and [Bing Wang](http://www.cs.ox.ac.uk/people/bing.wang/), Department of Computer Science, University of Oxford.

Please contact them (email: changhao.chen@cs.ox.ac.uk; bing.wang@cs.ox.ac.uk), if you have any question or would like to add your work on this list.
