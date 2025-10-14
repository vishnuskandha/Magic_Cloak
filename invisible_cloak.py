#!/usr/bin/env python3
"""
Invisible Cloak Effect using OpenCV
===================================

This script creates a Harry Potter-style invisible cloak effect using computer vision.
It detects red-colored objects in the video feed and replaces them with the background,
creating an invisibility effect.

Author: Vishnu Skandha (@vishnuskandha)
GitHub: https://github.com/vishnuskandha
Project: Magic Cloak Effect
License: MIT
Created: October 2025

Requirements:
    - OpenCV (cv2)
    - NumPy
    - Webcam/Camera

Usage:
    python invisible_cloak.py
    
Controls:
    - ESC key: Exit the program
    - Use a bright red cloth for best results
"""

import numpy as np
import cv2
import time
import sys

def initialize_camera():
    """Initialize camera and capture stable background."""
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    print("Please step out of frame for background capture...")
    time.sleep(3)
    
    print("Capturing background (30 frames)...")
    background = None
    for i in range(30):
        ret, frame = cap.read()
        if ret:
            background = frame
            print(f"Frame {i+1}/30", end='\r')
    
    print("\nBackground captured successfully!")
    print("Now wear your red cloak and step into the frame!")
    
    return cap, background


def create_invisibility_effect():
    """Main function to create the invisible cloak effect."""
    # Initialize camera and background
    cap, background = initialize_camera()
    
    # Create morphological kernels outside the loop for efficiency
    kernel = np.ones((3, 3), np.uint8)
    
    print("\nStarting invisible cloak effect...")
    print("Controls: Press ESC to exit")

    try:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                print("Error: Failed to read from camera")
                break
                
            # Convert to HSV color space for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Define range for bright red color (excludes skin tones)
            # Higher saturation and value to target bright red objects only
            lower_red1 = np.array([0, 150, 100])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            
            lower_red2 = np.array([170, 150, 100])
            upper_red2 = np.array([179, 255, 255])  # Fixed: HSV hue max is 179, not 180
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            # Combine both red masks
            red_mask = mask1 + mask2
            
            # Apply morphological operations to clean up the mask
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=3)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations=3)
            
            # Create inverse mask for non-red areas
            non_red_mask = cv2.bitwise_not(red_mask)
            
            # Apply masks to get final result
            background_part = cv2.bitwise_and(background, background, mask=red_mask)
            current_part = cv2.bitwise_and(img, img, mask=non_red_mask)
            
            # Combine both parts to create invisibility effect
            final_output = cv2.addWeighted(background_part, 1, current_part, 1, 0)
            
            # Display the result
            cv2.imshow("Invisible Cloak Effect - @vishnuskandha", final_output)
            
            # Break on ESC key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("\nExiting invisible cloak effect...")
                break
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete!")


if __name__ == "__main__":
    """Entry point of the program."""
    print("=" * 60)
    print("INVISIBLE CLOAK EFFECT")
    print("=" * 60)
    print("Created by: Vishnu Skandha (@vishnuskandha)")
    print("GitHub: https://github.com/vishnuskandha")
    print("=" * 60)
    
    try:
        create_invisibility_effect()
    except Exception as e:
        print(f"Failed to start program: {e}")
        sys.exit(1)
    
    