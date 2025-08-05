import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os


class BubbleAnalyzer:
    """analyzes foam bubble areas (after preprocessing) frame by frame in a video, and stores data for plotting."""
    
    def __init__(self, video_path, output_path="contour_areas.xlsx"):
        self.video_path = video_path
        self.output_path = output_path
        self.all_frame_areas = []   # list of lists, contains areas of all frames
        
        # Configurable parameters
        self.crop_y1, self.crop_y2 = 200, 400          # frame cropping, pixel indices
        self.crop_x1, self.crop_x2 = 650, 760
        self.gaussian_kernel_size = 5
        self.threshold_value = 75                      # for OTSU
        self.min_area, self.max_area = 50, 400         # for filtering out area values
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame: crop, grayscale, gaussian blur, and otsu threshold."""
        # Crop the frame
        cropped = frame[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
        
        # grayscaling
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blurring
        blurred = cv2.GaussianBlur(gray, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
        
        # Apply OTSU threshold
        _, thresholded = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        return thresholded, cropped
    
    def extract_contour_areas(self, thresholded):
        """Extract all bubble contour areas from a thresholded frame."""
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = []    # to store the areas of a single frame
        for contour in contours:
            area = cv2.contourArea(contour)
            areas.append(area)
        return areas, contours
    
    def process_video(self, show_preview=True):
        """Process the entire video and extract bubble areas from each frame."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Error opening video file:", self.video_path)
            return False
        
        frame_count = 0
        print("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Preprocess frame
                thresholded, cropped = self.preprocess_frame(frame)
                
                # Extract contours and areas
                areas, contours = self.extract_contour_areas(thresholded)
                self.all_frame_areas.append(areas)
                
                # Optional preview
                if show_preview:
                    contour_img = cropped.copy()
                    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
                    cv2.imshow('Bubble Contours', contour_img)
                    cv2.imshow('Thresholded', thresholded)

                    # stop by pressing q
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing stopped by user")
                        break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")
            
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Video processing completed. Total frames: {frame_count}")
        return True
    
    def save_data(self):
        """Save the extracted area data to an Excel file."""
        if len(self.all_frame_areas) == 0:
            print("No data to save. Please process video first.")
            return False
        
        try:
            # Find the maximum number of contours in any frame, needed so that rest frames data can be padded
            max_len = 0
            for areas in self.all_frame_areas:
                if len(areas) > max_len:
                    max_len = len(areas)
            
            if max_len == 0:
                print("No contours found in any frame")
                return False
            
            # Pad areas with NaN values to make all lists the same length
            padded_areas = []
            for areas in self.all_frame_areas:
                padded = areas.copy()
                # Add NaN values to make all lists the same length
                while len(padded) < max_len:
                    padded.append(np.nan)
                padded_areas.append(padded)
            
            # Create DataFrame (transpose so each frame is a column)
            df = pd.DataFrame(padded_areas).T
            
            # Name columns F1, F2, F3, etc.
            column_names = []
            for i in range(df.shape[1]):
                column_names.append(f"F{i+1}")
            df.columns = column_names
            
            # Sort areas within each frame
            for column in df.columns:
                df[column] = df[column].sort_values(ascending=True).reset_index(drop=True)
            
            # Save to Excel
            df.to_excel(self.output_path, index=False)
            print(f"Data saved to {self.output_path}")
            return True
        
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def get_summary_stats(self):
        """Get overall statistics for entire video."""
        if len(self.all_frame_areas) == 0:
            return {}
        
        # Flatten all areas into one list
        all_areas = []
        for frame_areas in self.all_frame_areas:
            for area in frame_areas:
                all_areas.append(area)

        # filter out area values, to neglect the inaccurate contours
        filtered_areas = []
        for area in all_areas:
            if self.min_area <= area <= self.max_area:
                filtered_areas.append(area)
        
        # Calculate statistics
        total_frames = len(self.all_frame_areas)
        total_bubbles = len(all_areas)
        filtered_bubbles = len(filtered_areas)
        avg_bubbles_per_frame = total_bubbles / total_frames if total_frames > 0 else 0
        
        area_min = min(all_areas) if all_areas else 0
        area_max = max(all_areas) if all_areas else 0
        
        filtered_min = min(filtered_areas) if filtered_areas else 0
        filtered_max = max(filtered_areas) if filtered_areas else 0
        
        stats = {
            'total_frames': total_frames,
            'total_bubbles': total_bubbles,
            'filtered_bubbles': filtered_bubbles,
            'avg_bubbles_per_frame': avg_bubbles_per_frame,
            'area_min': area_min,
            'area_max': area_max,
            'filtered_area_min': filtered_min,
            'filtered_area_max': filtered_max
        }
        
        return stats


class BubbleVisualizer:
    """A class for visualizing bubble area data."""
    
    def __init__(self, data_path, min_area=50, max_area=400):
        self.data_path = data_path
        self.min_area = min_area
        self.max_area = max_area
        self.data = None
        self.filtered_data = None
        
    def load_data(self):
        """Load data from Excel file."""
        try:
            self.data = pd.read_excel(self.data_path)
            # Apply area filtering
            self.filtered_data = self.data.copy()
            
            # Filter out values outside the range
            for column in self.filtered_data.columns:
                mask = (self.filtered_data[column] >= self.min_area) & (self.filtered_data[column] <= self.max_area)
                self.filtered_data.loc[~mask, column] = np.nan
            
            print(f"Data loaded from {self.data_path}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def plot_static_overview(self):
        """Create static plots showing data overview."""
        if self.filtered_data is None:
            print("No data loaded. Please load data first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Bubble Area Analysis Overview', fontsize=16)
        
        # Overall distribution
        all_areas = []
        for column in self.filtered_data.columns:
            for value in self.filtered_data[column]:
                if not pd.isna(value):
                    all_areas.append(value)
        
        axes[0, 0].hist(all_areas, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Overall Area Distribution across all frames')
        axes[0, 0].set_xlabel('Area')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot across sample frames (every 10th frame to avoid overcrowding)
        sample_columns = []
        sample_labels = []
        step = max(1, len(self.filtered_data.columns) // 10)
        
        for i in range(0, len(self.filtered_data.columns), step):
            column_name = self.filtered_data.columns[i]
            column_data = self.filtered_data[column_name].dropna()
            if len(column_data) > 0:
                sample_columns.append(column_data)
                sample_labels.append(f'F{i+1}')
        
        if sample_columns:
            axes[0, 1].boxplot(sample_columns, labels=sample_labels)
            axes[0, 1].set_title('Area Distribution Across Sample Frames')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('Area')
            for tick in axes[0, 1].get_xticklabels():
                tick.set_rotation(45)
        
        # Mean area over time
        mean_areas = []
        for column in self.filtered_data.columns:
            mean_val = self.filtered_data[column].mean()
            mean_areas.append(mean_val)
        
        frame_numbers = list(range(1, len(mean_areas) + 1))
        axes[1, 0].plot(frame_numbers, mean_areas)
        axes[1, 0].set_title('Mean Bubble Area Over Time')
        axes[1, 0].set_xlabel('Frame Number')
        axes[1, 0].set_ylabel('Mean Area')
        
        # Bubble count over time
        bubble_counts = []
        for column in self.filtered_data.columns:
            count = self.filtered_data[column].count()
            bubble_counts.append(count)
        
        axes[1, 1].plot(frame_numbers, bubble_counts)
        axes[1, 1].set_title('Number of Bubbles Over Time')
        axes[1, 1].set_xlabel('Frame Number')
        axes[1, 1].set_ylabel('Bubble Count')
        
        plt.tight_layout()
        plt.show()
    
    def create_animated_histogram(self, interval=100):
        """Create animated histogram showing area distribution over time."""
        if self.filtered_data is None:
            print("No data loaded. Please load data first.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate dynamic y-axis limit
        max_count = 0
        for column in self.filtered_data.columns:
            current_data = self.filtered_data[column].dropna()
            if len(current_data) > 0:
                hist, _ = np.histogram(current_data, bins=30)
                if hist.max() > max_count:
                    max_count = hist.max()
        
        data_min = self.filtered_data.min().min()
        data_max = self.filtered_data.max().max()
        
        def init():
            ax.clear()
            ax.set_title('Bubble Area Distribution Over Time')
            ax.set_xlabel('Area')
            ax.set_ylabel('Frequency')
            ax.set_xlim(data_min, data_max)
            ax.set_ylim(0, max_count * 1.1)
            return []
        
        def update(frame_idx):
            ax.clear()
            ax.set_title(f'Bubble Area Distribution - Frame {frame_idx + 1}')
            ax.set_xlabel('Area')
            ax.set_ylabel('Frequency')
            ax.set_xlim(data_min, data_max)
            ax.set_ylim(0, max_count * 1.1)
            
            column_name = self.filtered_data.columns[frame_idx]
            current_data = self.filtered_data[column_name].dropna()
            
            if len(current_data) > 0:
                sns.histplot(current_data, ax=ax, bins=30, kde=True, color='blue')
                
                # Add statistics text
                mean_val = current_data.mean()
                std_val = current_data.std()
                count_val = len(current_data)
                
                stats_text = f'Count: {count_val}\nMean: {mean_val:.1f}\nStd: {std_val:.1f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            return []
        
        num_frames = self.filtered_data.shape[1]
        ani = FuncAnimation(fig, update, frames=range(num_frames), 
                           init_func=init, repeat=False, interval=interval)
        
        plt.tight_layout()
        plt.show()
        return ani


# Main execution functions
def main_analysis(video_path, output_path="contour_areas_file.xlsx", show_preview=True):
    """Main function to run the complete bubble analysis."""
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
    
    # Initialize analyzer
    analyzer = BubbleAnalyzer(video_path, output_path)
    
    # Process video
    if not analyzer.process_video(show_preview):
        return False
    
    # Save data
    if not analyzer.save_data():
        return False
    
    # Print summary statistics
    stats = analyzer.get_summary_stats()
    print("Analysis Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True


def main_visualization(data_path="contour_areas.xlsx"):
    """Main function to run the visualization."""
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return False
    
    # Initialize visualizer
    visualizer = BubbleVisualizer(data_path)
    
    # Load data
    if not visualizer.load_data():
        return False
    
    # Create visualizations
    visualizer.plot_static_overview()
    
    # Create animated visualization
    print("Creating animated histogram... Close the window when done viewing.")
    ani = visualizer.create_animated_histogram(interval=50)
    
    return True


# Example usage
if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "60mlpermin.mp4"
    OUTPUT_PATH = "contour_areas_data.xlsx"
    
    # Run analysis
    print("=== BUBBLE AREA ANALYSIS ===")
    if main_analysis(VIDEO_PATH, OUTPUT_PATH, show_preview=True):
        print("\n=== VISUALIZATION ===")
        main_visualization(OUTPUT_PATH)
    else:
        print("Analysis failed. Please check the error messages above.")
