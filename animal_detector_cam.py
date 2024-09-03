import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('animal_species_detector.h5')

# Define labels (make sure these match the class indices used during training)
class_labels = list(train_generator.class_indices.keys())

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    resized_frame = cv2.resize(frame, (img_width, img_height))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Display the result on the frame
    label = f"{class_labels[predicted_class]}: {confidence * 100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Animal Species Detector', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close the window
cap.release()
cv2.destroyAllWindows()
