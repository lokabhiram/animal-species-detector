# Animal Species Detection using CNN

This project is a Convolutional Neural Network (CNN) model built with TensorFlow and Keras to classify different animal species. The model is trained on images and can predict the species of animals in new images.

## Model Architecture

The model is a simple Convolutional Neural Network (CNN) with the following layers:
- Conv2D layers followed by MaxPooling2D layers.
- Flatten layer to convert the 2D matrix to a vector.
- Dense layers with ReLU activation.
- Dropout layer for regularization.
- Final Dense layer with softmax activation for classification.

## Project Structure

- `animal_species_detector.h5`: The trained model file.
- `train.ipynb`: Script to train the model.
- `predict.py`: Script to make predictions using the trained model.
- `raw-img/`: Directory containing the training images, organized by class.
- `README.md`: This file.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/lokabhiram/animal-species-detector.git
    cd animal-species-detector
    ```

2. **Create a virtual environment and install dependencies**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Prepare your dataset**:
    - Place your dataset in the `raw-img/` directory, organized into subdirectories by class (e.g., `raw-img/lion`, `raw-img/tiger`, etc.).

## Training the Model

To train the model, run all cells in train.ipynb

## Predict animal species

To predict the animal species of an image, run:

```python
python predict.py
```
## Contributors

- [Lokabhiram](https://github.com/lokabhiram)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
