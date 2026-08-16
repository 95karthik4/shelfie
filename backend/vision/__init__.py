"""Local vision stage: spine detection + quality-gated OpenCV fallback.

Pure Python + ultralytics + opencv-python. Zero Django imports. Nothing here
talks to the network except the one-time model weight download ultralytics
performs on first load.
"""
