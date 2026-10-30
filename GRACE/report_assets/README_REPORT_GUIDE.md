# EdgeFireUAV — Project Report Visual Assets & Empirical Proofs Guide

This folder (`report_assets/`) contains all the diagrams, quantitative performance curves, dataset visualizations, batch validation predictions, and real-world aerial test images required for your academic thesis, conference paper, or project report.

Every image has been renamed to indicate its purpose and logical position in a technical report structure.

---

## Quick Reference Summary Table

| Index | File Name | Suggested Report Section | Key Technical Proof / Purpose |
|---|---|---|---|
| **01** | `01_system_architecture_diagram.png` | **Ch. 3: System Architecture** | Full UAV sensing, Jetson Nano compute, dual AI engine, geolocation, and GCS telemetry flow. |
| **02** | `02_posture_classification_and_triage_flowchart.png` | **Ch. 3: Methodology / Triage** | Skeletal joint kinematics + aerial aspect ratio decision logic and the mathematical rescue triage formula. |
| **03** | `03_edge_jetson_tensorrt_deployment_pipeline.png` | **Ch. 3 / 4: Edge Implementation** | 3-stage model optimization: PyTorch (.pt) → ONNX Opset 12 → TensorRT FP16 (.engine) with latency benchmark. |
| **04** | `04_model_training_and_validation_loss_convergence.png` | **Ch. 5: Model Training & Validation** | Loss curves (box, cls, dfl) and metric curves (precision, recall, mAP50, mAP50-95) across training epochs. |
| **05** | `05_normalized_confusion_matrix_posture_classes.png` | **Ch. 5: Model Evaluation** | Class-normalized accuracy across Lying, Sitting, and Standing classes proving high discriminative power. |
| **06** | `06_raw_counts_confusion_matrix.png` | **Ch. 5: Model Evaluation** | Unnormalized true positive / false positive instance counts over the validation split. |
| **07** | `07_precision_recall_curve_mAP50.png` | **Ch. 5: Model Evaluation** | Precision-Recall curve proving mAP@0.5 = **99.5%** across all posture classes. |
| **08** | `08_f1_score_vs_confidence_threshold_curve.png` | **Ch. 5: Model Evaluation** | F1-Score vs. Confidence curve identifying the optimal operating decision threshold (peak F1 = 0.99 at conf ~0.65). |
| **09** | `09_precision_vs_confidence_curve.png` | **Ch. 5: Model Evaluation** | Precision behavior as detection confidence threshold increases towards 1.0. |
| **10** | `10_recall_vs_confidence_curve.png` | **Ch. 5: Model Evaluation** | Recall stability across detection confidence thresholds. |
| **11** | `11_posture_dataset_class_distribution_and_spatial_density.jpg` | **Ch. 4: Dataset Preparation** | Class instance counts and spatial scatter density of victim bounding boxes in the normalized coordinate frame. |
| **12** | `12_mosaic_data_augmentation_training_batch_sample.jpg` | **Ch. 4: Training Methodology** | 4-image Mosaic data augmentation, scaling, translation, and affine shifts preventing overfitting. |
| **13** | `13_validation_batch_ground_truth_posture_annotations.jpg` | **Ch. 5: Qualitative Validation** | Ground truth bounding boxes and class labels from the validation partition. |
| **14** | `14_validation_batch_model_posture_predictions.jpg` | **Ch. 5: Qualitative Validation** | Direct comparison showing model predictions matching ground truth labels with high confidence. |
| **15** | `15_raw_aerial_uav_test_scene_1.jpg` | **Ch. 5: Field Testing & Inference** | Unprocessed raw aerial drone capture showing smoke, fire zones, and persons in the disaster area. |
| **16** | `16_annotated_aerial_detection_fire_smoke_and_posture_scene_1.jpeg` | **Ch. 5: Field Testing & Inference** | Dual model inference result showing simultaneous fire/smoke detection and victim posture classification. |
| **17** | `17_raw_aerial_uav_test_scene_2.jpg` | **Ch. 5: Field Testing & Inference** | Unprocessed raw aerial drone capture of victims in open terrain. |
| **18** | `18_annotated_aerial_detection_victim_postures_scene_2.jpeg` | **Ch. 5: Field Testing & Inference** | Model detection output on test scene 2 classifying victim postures with confidence percentages. |
| **19** | `19_realtime_video_inference_hud_triage_dashboard_frame.jpg` | **Ch. 5: Real-Time Edge Deployment** | Full HUD tactical display showing live FPS, posture counters (Lying, Sitting, Standing), and fire status. |
| **20** | `20_temporal_tracking_sequence_early_frame_100.jpg` | **Ch. 5: Video Inference & Tracking** | Temporal tracking proof at frame 100 demonstrating continuous victim localization under UAV motion. |
| **21** | `21_temporal_tracking_sequence_mid_frame_250.jpg` | **Ch. 5: Video Inference & Tracking** | Temporal tracking proof at frame 250 verifying posture stability across changing drone camera angles. |
| **22** | `22_temporal_tracking_sequence_late_frame_400.jpg` | **Ch. 5: Video Inference & Tracking** | Temporal tracking proof at frame 400 confirming persistent tracking and absence of false alarms. |

---

## Detailed Report Section Walkthrough & Ready-to-Use Captions

### SECTION 1: SYSTEM ARCHITECTURE & METHODOLOGY (Chapter 3)

#### Figure 1: End-to-End System Architecture
- **File:** `01_system_architecture_diagram.png`
- **Caption:** *Figure 3.1: Complete End-to-End Architecture of the EdgeFireUAV System, detailing onboard sensing (CSI Camera, Pixhawk MAVLink), Jetson Nano edge processing, dual TensorRT inference engines, ray-casting geolocation, and ground control station telemetry dispatch.*
- **Discussion in Report:** Highlight how the system eliminates heavy cloud dependency by performing all neural network inference, ray-casting GPS projection, and triage calculations directly onboard the Jetson Nano at low power (5–10W).

#### Figure 2: Posture Kinematics & Rescue Triage Decision Flow
- **File:** `02_posture_classification_and_triage_flowchart.png`
- **Caption:** *Figure 3.2: Multi-modal victim posture classification and rescue triage prioritization workflow, combining skeletal joint kinematics (COCO-17 keypoints), aerial bounding box aspect-ratio heuristics, and distance-weighted triage score formulation.*
- **Discussion in Report:** Explain the two complementary posture branches:
  1. *Kinematic Joint Angles:* Torso inclination and knee flexion angle $\theta_{knee} = \angle(\text{Hip}, \text{Knee}, \text{Ankle})$.
  2. *Aerial Aspect Ratio:* Ratio $R = h / w$, where $R \ge 1.35$ indicates standing, $0.75 \le R < 1.35$ indicates sitting, and $R < 0.75$ indicates lying (critical).
  3. *Triage Equation:* $P_{victim} = w_{posture} S_{posture} + w_{dist} \left(1 - \frac{d_{fire}}{d_{max}}\right) + w_{fire} S_{fire}$.

#### Figure 3: Edge Jetson Nano Deployment & Latency Benchmark
- **File:** `03_edge_jetson_tensorrt_deployment_pipeline.png`
- **Caption:** *Figure 3.3: Edge model compilation pipeline from PyTorch (.pt) through ONNX Opset 12 to TensorRT FP16 (.engine), alongside execution latency and frame rate comparisons across deployment runtimes.*
- **Discussion in Report:** Contrast the 3 FPS PyTorch CPU baseline with the 15–20 FPS achieved by TensorRT FP16 engine on the Jetson Nano's 128-core Maxwell GPU.

---

### SECTION 2: DATASET & TRAINING METHODOLOGY (Chapter 4)

#### Figure 4: Dataset Class Distribution & Spatial Bounding Box Density
- **File:** `11_posture_dataset_class_distribution_and_spatial_density.jpg`
- **Caption:** *Figure 4.1: Dataset distribution analysis depicting instance counts per posture class (Lying, Sitting, Standing) and 2D spatial scatter distribution of normalized bounding box centroids.*
- **Discussion in Report:** Discuss how the dataset was curated from the Atomic subset, consolidating ambulatory states (walking, running, standing) into Class 2 ("Standing"), separating "Sitting" (Class 1) and "Lying" (Class 0), and discarding degenerate boxes.

#### Figure 5: Mosaic Data Augmentation Sample
- **File:** `12_mosaic_data_augmentation_training_batch_sample.jpg`
- **Caption:** *Figure 4.2: Four-image mosaic training batch with dynamic affine transforms, scaling, and horizontal flips, bolstering network robustness against aerial scale variations and viewpoint changes.*

---

### SECTION 3: QUANTITATIVE EXPERIMENTAL RESULTS (Chapter 5)

#### Key Metric Numbers to Quote in Your Report:
- **Precision ($P$):** 99.3%
- **Recall ($R$):** 99.9%
- **mAP@0.50:** 99.5%
- **mAP@0.50:0.95:** 79.9%
- **Target Base Architecture:** YOLOv8n (nano, 3.2M parameters)
- **ONNX Export Size:** 12.2 MB

#### Figure 6: Model Training Convergence & Loss Curves
- **File:** `04_model_training_and_validation_loss_convergence.png`
- **Caption:** *Figure 5.1: Training and validation loss curves (bounding box loss, classification loss, distribution focal loss) and performance metrics (Precision, Recall, mAP50, mAP50-95) across training epochs.*
- **Discussion in Report:** Point out the steady convergence of both training and validation losses without evidence of severe overfitting, reaching plateau around epoch 35–45.

#### Figure 7: Normalized Confusion Matrix
- **File:** `05_normalized_confusion_matrix_posture_classes.png`
- **Caption:** *Figure 5.2: Normalized confusion matrix evaluating the 3-class victim posture detector on the unseen validation dataset.*
- **Discussion in Report:** Note the clean diagonal entries indicating high class separability. Mention that Lying and Standing classes exhibit near zero mutual confusion due to their distinct aspect-ratio and limb orientation signatures.

#### Figure 8: Precision-Recall (PR) Curve
- **File:** `07_precision_recall_curve_mAP50.png`
- **Caption:** *Figure 5.3: Precision-Recall (PR) curve for all posture classes at IoU threshold = 0.50, demonstrating an overall mAP of 99.5%.*

#### Figure 9: F1-Score vs. Confidence Threshold Curve
- **File:** `08_f1_score_vs_confidence_threshold_curve.png`
- **Caption:** *Figure 5.4: F1-Score as a function of classification confidence threshold, showing maximum F1 = 0.99 at an optimal decision threshold of approximately 0.65.*

#### Figure 10: Validation Ground Truth vs. Model Prediction Comparison
- **Files:** `13_validation_batch_ground_truth_posture_annotations.jpg` & `14_validation_batch_model_posture_predictions.jpg`
- **Caption:** *Figure 5.5: Side-by-side comparison between (a) ground truth human annotations and (b) YOLOv8n posture predictions on validation batch 0, confirming tight bounding box localization and accurate class assignment.*

---

### SECTION 4: REAL-WORLD TESTING & QUALITATIVE INFERENCE (Chapter 5 / 6)

#### Figure 11: Dual Fire/Smoke and Victim Detection in Aerial Scene 1
- **Files:** `15_raw_aerial_uav_test_scene_1.jpg` & `16_annotated_aerial_detection_fire_smoke_and_posture_scene_1.jpeg`
- **Caption:** *Figure 6.1: Dual-model aerial inference evaluation on Scene 1: (a) raw UAV capture; (b) processed output displaying fire/smoke boundary detection alongside localized victim postures and confidence scores.*

#### Figure 12: Victim Posture Classification in Open Terrain Scene 2
- **Files:** `17_raw_aerial_uav_test_scene_2.jpg` & `18_annotated_aerial_detection_victim_postures_scene_2.jpeg`
- **Caption:** *Figure 6.2: Victim posture classification on Scene 2: (a) raw aerial image; (b) detected victims correctly identified as Sitting and Standing with high confidence.*

#### Figure 13: Real-Time Tactical HUD & Posture Triage Dashboard
- **File:** `19_realtime_video_inference_hud_triage_dashboard_frame.jpg`
- **Caption:** *Figure 6.3: Real-time tactical head-up display (HUD) rendered by test_video_inference.py, showing live frame index, FPS counter, victim posture tallies (Lying: Critical, Sitting: Moderate, Standing: Ambulatory), and active fire status.*

#### Figure 14: Temporal Video Tracking Sequence Across Flight Trajectory
- **Files:** `20_temporal_tracking_sequence_early_frame_100.jpg`, `21_temporal_tracking_sequence_mid_frame_250.jpg`, `22_temporal_tracking_sequence_late_frame_400.jpg`
- **Caption:** *Figure 6.4: Multi-frame temporal tracking sequence (Frames 100, 250, and 400) demonstrating persistent victim localization and continuous posture estimation under simulated UAV flight dynamics.*
