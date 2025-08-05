# Foam Bubble Area Analysis

A Python-based computer vision tool for analyzing foam bubble areas in video recordings. This project extracts bubble contours from video frames, measures their areas, and provides comprehensive visualizations of bubble size distributions over time.

## Overview

This tool is designed for researchers and engineers studying foam dynamics, bubble formation, or fluid mechanics. It processes video files to:

- Detect and track bubble contours in real-time
- Measure bubble areas across all video frames
- Filter out noise and artifacts based on size criteria
- Generate comprehensive statistical analysis
- Create animated visualizations showing temporal changes

## Features

### Core Analysis

- **Automated bubble detection** using OpenCV contour detection
- **Configurable image preprocessing** (cropping, thresholding, filtering)
- **Area measurement** for each detected bubble
- **Data export** to Excel format for further analysis
- **Real-time preview** during processing

### Visualization & Analysis

- **Static overview plots**: Distribution histograms, box plots, trend analysis
- **Animated histograms**: Real-time area distribution changes
- **Statistical summaries**: Bubble counts, size ranges, temporal trends
- **Customizable filtering** to remove noise and artifacts

## Requirements

### Python Libraries

```
opencv-python >= 4.5.0
numpy >= 1.19.0
pandas >= 1.3.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
openpyxl >= 3.0.0  # For Excel file support
```

### System Requirements

- Python 3.7 or higher
- Sufficient RAM for video processing (depends on video size)
- Display capability for visualization (optional for headless processing)

## Installation

1. **Clone or download** the project files
2. **Install required packages**:
   ```bash
   pip install opencv-python numpy pandas matplotlib seaborn openpyxl
   ```
3. **Place your video file** in the same directory as the script

## Quick Start

### Basic Usage

```python
# Run complete analysis pipeline
if __name__ == "__main__":
    VIDEO_PATH = "your_video.mp4"
    OUTPUT_PATH = "bubble_analysis.xlsx"

    # Process video and analyze bubbles
    if main_analysis(VIDEO_PATH, OUTPUT_PATH, show_preview=True):
        # Create visualizations
        main_visualization(OUTPUT_PATH)
```

### Step-by-Step Usage

```python
# 1. Initialize analyzer
analyzer = BubbleAnalyzer("foam_video.mp4", "results.xlsx")

# 2. Process video
analyzer.process_video(show_preview=True)

# 3. Save data
analyzer.save_data()

# 4. Get statistics
stats = analyzer.get_summary_stats()
print(stats)

# 5. Visualize results
visualizer = BubbleVisualizer("results.xlsx")
visualizer.load_data()
visualizer.plot_static_overview()
visualizer.create_animated_histogram()
```

## Configuration

### BubbleAnalyzer Parameters

```python
analyzer = BubbleAnalyzer(video_path, output_path)

# Customize detection parameters:
analyzer.crop_y1, analyzer.crop_y2 = 200, 400    # Vertical crop region
analyzer.crop_x1, analyzer.crop_x2 = 650, 760    # Horizontal crop region
analyzer.gaussian_kernel_size = 5                 # Blur kernel size
analyzer.threshold_value = 75                     # Binary threshold
analyzer.min_area, analyzer.max_area = 50, 400    # Size filter range
```

### Key Parameters Explained

| Parameter              | Description                           | Typical Range   | Impact                            |
| ---------------------- | ------------------------------------- | --------------- | --------------------------------- |
| `crop_region`          | Region of interest in video frame     | Video-dependent | Focuses analysis on foam area     |
| `threshold_value`      | Binary threshold for bubble detection | 60-90           | Higher = more selective detection |
| `gaussian_kernel_size` | Blur kernel size (must be odd)        | 3, 5, 7         | Larger = more noise reduction     |
| `area_filter_range`    | Min/max bubble areas to include       | (10, 1000)      | Removes noise and large artifacts |

## Output Files

### Excel Data File

- **Structure**: Each column represents one video frame
- **Content**: Bubble areas sorted in ascending order within each frame
- **Format**: Padded with NaN for consistent column lengths
- **Usage**: Import into any statistical software for further analysis

### Console Output

```
Analysis Summary:
  total_frames: 500
  total_bubbles: 12847
  filtered_bubbles: 11203
  avg_bubbles_per_frame: 25.69
  area_min: 12.5
  area_max: 2847.0
  filtered_area_min: 50.0
  filtered_area_max: 400.0
```

## Visualization Types

### 1. Static Overview (4-panel plot)

- **Overall Distribution**: Histogram of all bubble areas
- **Frame Sampling**: Box plots across representative frames
- **Temporal Trends**: Mean area and bubble count over time

### 2. Animated Histogram

- **Real-time visualization** of area distribution changes
- **Frame-by-frame statistics** overlay
- **Configurable playback speed**

### Common Issues

**Video Won't Open**

```
Error opening video file: your_video.mp4
```

- Check file path and name
- Ensure video format is supported (MP4, AVI, MOV)
- Verify file isn't corrupted

**No Bubbles Detected**

```
No contours found in any frame
```

- Adjust `threshold_value` (try 60-90 range)
- Check crop region covers foam area
- Verify video has visible bubbles

**Memory Issues**

- Process shorter video segments
- Reduce video resolution
- Increase system RAM

**Poor Detection Quality**

- Fine-tune `threshold_value`
- Adjust `gaussian_kernel_size`
- Modify crop region
- Check lighting conditions in video

### Parameter Tuning Guide

1. **Start with crop region**: Ensure it covers only the foam area
2. **Adjust threshold**: Lower for more detection, higher for less noise
3. **Set size filters**: Remove unrealistic bubble sizes
4. **Test on short clips** before processing full videos

## Project Structure

```
foam-bubble-analyzer/
├── bubble_analyzer.py          # Main analysis code
├── README.md                   # This file
├── 60mlpermin.mp4             # Input video file
├── contour_areas.xlsx         # Output data file
└── requirements.txt           # Python dependencies
```

## Technical Details

### Image Processing Pipeline

1. **Frame Extraction**: Read video frame by frame
2. **Region Cropping**: Focus on area of interest
3. **Grayscale Conversion**: Reduce computational complexity
4. **Gaussian Blur**: Reduce noise while preserving edges
5. **Binary Thresholding**: Create binary image for contour detection
6. **Contour Detection**: Find bubble boundaries
7. **Area Calculation**: Measure each bubble's area

### Data Processing

- **Temporal Organization**: Areas organized by frame
- **Size Filtering**: Remove artifacts based on realistic bubble sizes
- **Statistical Analysis**: Frame-wise and overall statistics
- **Data Export**: Structured format for external analysis
