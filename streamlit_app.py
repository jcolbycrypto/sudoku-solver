import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to display the Sudoku grid
def display_sudoku_grid(grid):
    st.write("Sudoku Puzzle:")
    for row in grid:
        st.write(row)

# Function to process the uploaded image and extract the Sudoku grid
def process_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to detect the largest square (the Sudoku puzzle)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Check if we have four corners (for a valid Sudoku grid)
        if len(approx) == 4:
            # Extract points from the contour
            points = np.array([point[0] for point in approx], dtype="float32")
            print("Extracted Points:", points)  # Debugging statement
            
            # Order the points: top-left, top-right, bottom-right, bottom-left
            top_left = points[np.argmin(points.sum(axis=1))]
            bottom_right = points[np.argmax(points.sum(axis=1))]
            top_right = points[np.argmin(np.diff(points, axis=1))]
            bottom_left = points[np.argmax(np.diff(points, axis=1))]
            
            ordered_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
            print("Ordered Points:", ordered_points)  # Debugging statement
            
            # Create destination points for the top-down view of the Sudoku puzzle
            width = height = 450  # Fixed size for Sudoku grid
            dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
            
            # Warp the perspective to get a top-down view of the Sudoku puzzle
            matrix = cv2.getPerspectiveTransform(ordered_points, dst)  # Perspective transform matrix
            warped = cv2.warpPerspective(image, matrix, (width, height))
            
            return warped
    return None

# Streamlit App
st.title("Sudoku Solver")

# Upload the Sudoku puzzle image
uploaded_image = st.file_uploader("Upload a Sudoku puzzle image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Open the image using PIL
    image = Image.open(uploaded_image)
    
    # Convert the image to RGB if it's not already in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert the PIL image to a NumPy array (for OpenCV)
    image = np.array(image)

    # Process the image to extract the Sudoku grid
    grid = process_image(image)
    
    if grid is not None:
        display_sudoku_grid(grid)
        
        if st.button("Solve Puzzle"):
            if solve_sudoku(grid):
                st.write("Solved Sudoku Puzzle:")
                display_sudoku_grid(grid)
            else:
                st.write("The puzzle could not be solved.")
    else:
        st.write("Could not detect a valid Sudoku grid in the image.")
