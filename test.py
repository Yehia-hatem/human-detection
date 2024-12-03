from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

try:
    # Load the trained YOLOv8 model
    print("Loading model...")
    model = YOLO("../Hcrowd/Hcrowded_project.pt")  # Path to your trained model
    print("Model loaded successfully!")

    # Run inference on a test image
    print("Running inference on image...")
    results = model.predict("../Hcrowd/test2.jpg")  # Replace with your test image path
    print("Inference completed!")

    # Since `results` is a list, access the first result (as there could be multiple)
    result = results[0]
    
    # Define the output directory and ensure it exists
    output_dir = "../Hcrowd/results"
    os.makedirs(output_dir, exist_ok=True)  # Create results directory if it doesn't exist
    
    # Define the output image path for saving the result
    output_image_path = os.path.join(output_dir, "test2.C.jpg")
    
    # Save the result image with bounding boxes
    result.save(output_image_path)  # Save the result in the results directory
    print(f"Results saved at {output_image_path}!")

    # Load the saved image with bounding boxes
    print(f"Loading image from {output_image_path}...")
    image = cv2.imread(output_image_path)

    if image is None:
        print(f"Error: Image not found at {output_image_path}")
    else:
        print(f"Image loaded successfully!")

    # Convert the image from BGR (OpenCV default) to RGB for Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes using OpenCV
    print("Displaying image... Press 'Esc' to close.")
    cv2.imshow("Detection Results", image)

    # Wait for the Esc key to be pressed
    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ASCII code for 'Esc'
            print("Esc key pressed. Closing...")
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")
