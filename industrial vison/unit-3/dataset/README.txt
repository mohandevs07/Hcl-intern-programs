Industrial Vision Unit 3 - Sample Dataset Pack

These are SYNTHETIC practice images generated for testing your scripts.
They are not real industrial inspection data and should not be used to claim real defect-detection performance.

Folder map:
classification/train and classification/validation:
  Use for Programs 11 (MobileNet) and 12 (ResNet).
  Classes: defective, non_defective.

yolo:
  Use for Program 13. Includes images, matching YOLO .txt bounding-box labels,
  and data.yaml. The yaml path is relative to the yolo folder.

segmentation:
  Use for Program 14. Each image has a same-named binary mask.

super_resolution:
  Use low_resolution.jpg for Program 15.
  reference_high_resolution.png is included only for visual comparison.

Important:
- Update the paths in your Python programs to point to these folders.
- Program 15 still needs a compatible pretrained ESPCN_x4.pb model file.
  This dataset pack does NOT include that model.
