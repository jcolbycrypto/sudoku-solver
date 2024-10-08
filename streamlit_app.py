import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Function to display the Sudoku grid
def display_sudoku_grid(grid):
    st.write("Sudoku Puzzle:")
    for row in grid:
        st.write(row)

# Backtracking algorithm to solve the Sudoku puzzle
def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

# Function to process the uploaded image and extract the Sudoku grid
def process_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to detect the largest square (the Sudoku puzzle)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Extract the Sudoku grid from the image
    if len(approx) == 4:
        points = np.float32([point[0] for point in approx])
        points = sorted(points, key=lambda x: (x[1], x[0]))
        top_left, top_right, bottom_left, bottom_right = points
        
        # Warp the perspective to get a top-down view of the Sudoku puzzle
        width = height = 450  # Dimensions for the new perspective
        dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(points, dst)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        
        # Split the warped image into a 9x9 grid
        cell_size = width // 9
        grid = np.zeros((9, 9), dtype=int)
        for i in range(9):
            for j in range(9):
                x, y = j * cell_size, i * cell_size
                cell = warped[y:y + cell_size, x:x + cell_size]
                cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                digit = pytesseract.image_to_string(cell_gray, config='--psm 10 digits')
                
                # If a digit is detected, place it in the grid
                try:
                    grid[i, j] = int(digit)
                except ValueError:
                    grid[i, j] = 0  # Empty cell
        return grid
    return None

# Function to load image from a URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

# Streamlit App
st.title("Sudoku Solver")

# Option to select image source
source = st.radio("Select image source", ("Upload Image", "Enter URL"))

if source == "Upload Image":
    # Upload the Sudoku puzzle image
    uploaded_image = st.file_uploader("Upload a Sudoku puzzle image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
elif source == "Enter URL":
    # Enter the image URL
    image_url = st.text_input("Enter image URL")

    if image_url:
        image = load_image_from_url(image_url)
else:
    image = None

# If an image is available, process it
if image is not None:
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
