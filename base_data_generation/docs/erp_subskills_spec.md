# ERP Capability and QA Generation Specification

This note describes the capability groups used when generating PanoWorld-style SFT and benchmark questions from ERP metadata. The goal is to turn object-level panorama annotations into reliable samples for full-surround spatial reasoning.

## 1. Basic Object Understanding

These samples test whether a model can identify, localize, and describe objects in an ERP panorama.

### 1.1 Object Identification

- **Rule**: sample salient objects with reliable semantic labels.
- **Question**: ask what object appears inside a specified region or bounding box.
- **Answer**: use the normalized object name or enriched semantic label.

### 1.2 Attribute Recognition

- **Rule**: use attributes from semantic enrichment, such as color, state, material, or count.
- **Question**: ask about an observable attribute of a target object.
- **Answer**: use the attribute field when confidence is high.

### 1.3 Region Captioning

- **Rule**: crop or reference a target object/region and use the enriched caption fields.
- **Question**: ask for a concise description of the region.
- **Answer**: use `caption_brief` or a filtered dense caption.

## 2. Omnidirectional ERP Reasoning

These samples test whether a model understands the panorama as a continuous 360-degree observation instead of a flat image.

### 2.1 Absolute Direction

- **Rule**: use ERP yaw/pitch coordinates of high-confidence entities.
- **Question**: ask whether an object is in front, behind, left, right, above, or below the observer.
- **Answer**: compute the answer from the object's spherical center.

### 2.2 Relative Direction

- **Rule**: choose two salient entities with clear yaw/pitch separation.
- **Question**: ask which object is to the left/right/above/below another object in the viewer-centered frame.
- **Answer**: compare their spherical centers.

### 2.3 Boundary Continuity

- **Rule**: find objects whose boxes or masks cross the left/right ERP boundary.
- **Question**: ask whether two edge fragments correspond to the same physical object.
- **Answer**: use the merged ERP instance ID.

### 2.4 Polar Distortion Awareness

- **Rule**: select objects near the top or bottom ERP poles where pixel stretching is strong.
- **Question**: ask how ERP distortion affects the apparent object shape.
- **Answer**: describe the real-world shape using metadata and object category priors.

## 3. 3D Spatial Understanding

These samples use depth and 3D point statistics to test whether a model reasons beyond the stretched 2D panorama plane.

### 3.1 Distance Estimation

- **Rule**: use `depth.median`, `depth.range_m`, or another robust depth statistic.
- **Question**: ask for the approximate distance from the camera to an object.
- **Answer**: output a numeric bucket or approximate metric distance.

### 3.2 3D Relative Position

- **Rule**: choose object pairs with sufficiently different depth values.
- **Question**: ask which object is physically closer to the camera.
- **Answer**: compare robust depth values after filtering uncertain pairs.

### 3.3 Real Size and Shape

- **Rule**: combine spherical bounding spans with depth to estimate physical size.
- **Question**: ask which size bucket best matches an object.
- **Answer**: use geometry-derived height/width estimates when confidence is high.

### 3.4 Spatial Layout and Cognitive Map

- **Rule**: select several stable landmarks from a room or scene.
- **Question**: ask about functional zones or landmark layout relative to the observer.
- **Answer**: derive the answer from object positions and scene-level metadata.

## 4. Sampling Strategy

To avoid combinatorial explosion and reduce noisy samples:

1. Select 5-10 salient objects per panorama using area, semantic confidence, and category filters.
2. Generate a small balanced set from each capability group.
3. Prefer rule-derived answers for benchmark samples.
4. Use VLM-generated open-ended samples mainly for SFT expansion.
5. Deduplicate questions within the same scene.

## 5. Reliability Levels

- `high`: answer is directly computed from geometry, depth, count, or a deterministic rule.
- `medium`: answer is rule-derived but sensitive to thresholds or boundary cases.
- `llm_medium`: answer depends on VLM-generated language and should be used for training augmentation rather than core benchmark scoring.

## 6. Filtering Rules

Recommended defaults:

- `semantic.confidence >= 0.45`
- `depth.valid_ratio >= 0.2` for 3D samples.
- Relative depth difference ratio at least `0.15` for closer/farther comparisons.
- Limit maximum samples per scene so dense scenes do not dominate the distribution.
- Keep the capability distribution balanced across basic, omnidirectional, and 3D tasks.

## 7. Additional Capability Directions

Two useful extensions are:

- **Boundary continuity consistency**: ask whether left/right ERP fragments belong to the same instance, not just whether a boundary object exists.
- **Uncertainty calibration**: include hard cases involving occlusion, small objects, or polar distortion and ask the model to answer conservatively when evidence is weak.
